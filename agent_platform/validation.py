"""Shared config validation.

The point of this module is that there is exactly one implementation of "is this
config usable", used by the API on write and by the worker on load. Two copies of
these rules is how a config gets accepted by the editor and then rejected by the
worker, and that shows up as "the UI said it saved but the call broke".

Each check returns a message naming the field and how to fix it, because these
strings are shown verbatim in the UI. "unknown tool 'get_wheather'" is a bug
report; "agent 'primary' lists tool 'get_wheather', which does not exist — check
the name in the Tools registry" is a fix.
"""


class ConfigValidationError(ValueError):
    """A config that cannot work, carrying one message per problem.

    Subclasses ValueError so existing call sites that catch ValueError around
    registry validation keep working unchanged.
    """

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("Invalid config:\n  " + "\n  ".join(self.problems))


def validate_agent(agent, tool_names, agent_ids, *, allow_unknown_tools=False):
    """Problems with one agent, given the tool and agent ids it may reference.

    `allow_unknown_tools` exists for the create-then-fix path: an agent can
    reference a tool that is being created in the same save, and a squad's handoff
    edge can name an agent created in the same request.
    """
    problems = []
    agent_id = agent.get("id")

    if not agent_id or not isinstance(agent_id, str):
        problems.append("agent id must be a non-empty string")
        return problems

    if not (agent.get("system_prompt") or "").strip():
        problems.append(
            f"agent '{agent_id}' has an empty system prompt — a small local model "
            "has no task definition to fall back on"
        )

    for tool_name in agent.get("tools", []):
        if tool_name not in tool_names and not allow_unknown_tools:
            problems.append(
                f"agent '{agent_id}' lists tool '{tool_name}', which is not in the "
                f"tool registry — known tools: {', '.join(sorted(tool_names)) or '(none)'}"
            )

    for target in agent.get("handoffs", []):
        if target == agent_id:
            # Not merely useless: a handoff tool to itself gets offered to the
            # model as a real option, and picking it costs a turn for no change.
            problems.append(
                f"agent '{agent_id}' hands off to itself — a handoff must name a "
                "different agent"
            )
        elif target not in agent_ids and not allow_unknown_tools:
            problems.append(
                f"agent '{agent_id}' hands off to '{target}', which does not exist — "
                f"known agents: {', '.join(sorted(agent_ids)) or '(none)'}"
            )

    for rule in agent.get("rules", []):
        if not isinstance(rule, str):
            problems.append(f"agent '{agent_id}' has a rule that is not a string")
        elif not rule.strip():
            problems.append(
                f"agent '{agent_id}' has an empty rule — an empty instruction is "
                "not a rule and only costs prompt tokens"
            )

    return problems


def validate_agent_payload(agent, existing_agents, existing_tools):
    """Validate an agent being written through the API, before it is stored.

    Run against the post-write world rather than the current one, so that an agent
    can hand off to another agent created in the same batch. The caller is
    expected to have already checked for a duplicate id.
    """
    problems = []
    agent_id = agent.get("id", "")

    if not agent_id or not isinstance(agent_id, str):
        return ["agent id must be a non-empty string"]
    if not agent_id.replace("_", "").replace("-", "").isalnum():
        # The id becomes a filename on export and a URL segment on lookup, so
        # anything with a slash or space breaks one or both.
        problems.append(
            f"agent id '{agent_id}' may only contain letters, digits, '-' and '_'"
        )

    tool_names = set(existing_tools)
    # A handoff target always needs a handoff tool, and the worker offers the
    # handoff tools the agent lists. Checking that here means a handoff cannot be
    # saved into a state where build_prompt would KeyError on the missing tool.
    for target in agent.get("handoffs", []):
        tool_names.add(f"handoff_to_{target}")

    problems.extend(validate_agent(
        agent, tool_names, set(existing_agents) | {agent_id},
    ))
    return problems


def validate_squad(squad, agent_ids):
    """Problems with one squad.

    The graph checks are the interesting ones: an unreachable agent looks fine
    until a call never routes to it, and a second entry agent is a contradiction
    that would otherwise be decided by whichever row loaded last.
    """
    problems = []
    squad_id = squad.get("id", "")

    entry = squad.get("entry_agent")
    if not entry:
        problems.append(
            f"squad '{squad_id}' has no entry agent — a call has to start somewhere"
        )
    elif entry not in agent_ids:
        problems.append(
            f"squad '{squad_id}' entry agent '{entry}' does not exist"
        )

    edges = squad.get("edges", [])
    if not edges and len(agent_ids) > 1:
        problems.append(
            f"squad '{squad_id}' has no handoff edges, so only the entry agent "
            "can ever be reached"
        )

    for edge in edges:
        src, dst = edge.get("from"), edge.get("to")
        if not src or not dst:
            problems.append(f"squad '{squad_id}' has an edge missing its source or target")
            continue
        for label, node in (("source", src), ("target", dst)):
            if node not in agent_ids:
                problems.append(
                    f"squad '{squad_id}' edge {label} '{node}' does not exist"
                )

    # Reachability from the entry agent. Reported as one problem listing the
    # orphans rather than one per orphan, which is far easier to act on when a
    # graph has several.
    if entry and entry in agent_ids:
        reachable = {entry}
        changed = True
        adjacency = {}
        for edge in edges:
            src, dst = edge.get("from"), edge.get("to")
            if src and dst:
                adjacency.setdefault(src, []).append(dst)
        while changed:
            changed = False
            for node in list(reachable):
                for nxt in adjacency.get(node, []):
                    if nxt in agent_ids and nxt not in reachable:
                        reachable.add(nxt)
                        changed = True
        orphans = sorted(a for a in agent_ids if a not in reachable)
        if orphans:
            problems.append(
                f"squad '{squad_id}': {', '.join(orphans)} cannot be reached from "
                f"entry agent '{entry}' — connect it with a handoff edge or remove it"
            )

    return problems


def validate_tool(name, spec):
    """Problems with one tool definition."""
    problems = []
    if not name or not isinstance(name, str):
        return ["tool name must be a non-empty string"]
    if not name.replace("_", "").replace("-", "").isalnum():
        problems.append(
            f"tool name '{name}' may only contain letters, digits, '-' and '_'"
        )
    if not (spec.get("description") or "").strip():
        # The description is the only thing the model uses to decide when to call
        # the tool. Empty means a tool the model has no reason to ever call.
        problems.append(
            f"tool '{name}' has no description — the description is the model's "
            "only signal for when to call it"
        )

    schema = spec.get("args_schema")
    if schema is not None and not isinstance(schema, dict):
        problems.append(
            f"tool '{name}': args_schema must be an object, got "
            f"{type(schema).__name__}"
        )
    elif isinstance(schema, dict):
        properties = schema.get("properties", schema)
        if not isinstance(properties, dict):
            problems.append(f"tool '{name}': args_schema.properties must be an object")
        else:
            for arg_name, arg_spec in properties.items():
                if not isinstance(arg_spec, dict):
                    problems.append(
                        f"tool '{name}' arg '{arg_name}': each argument must be an "
                        f"object, got {type(arg_spec).__name__}"
                    )
    return problems


def validate_config_snapshot(agents, tools):
    """Whole-config check, used by the worker after a reload.

    Same rules as the per-write validation, applied to the full set, so a config
    that was assembled by several writes is checked as the thing it actually is
    rather than as a series of individually-valid edits.
    """
    problems = []
    tool_names = set(tools)
    agent_ids = set(agents)
    for agent_id, agent in agents.items():
        problems.extend(validate_agent(
            dict(agent, id=agent.get("id", agent_id)),
            tool_names, agent_ids,
        ))
    return problems
