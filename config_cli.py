#!/usr/bin/env python3
"""Create, edit, list and validate the agent configs in agent_platform/agents/.

The configs are plain JSON, but they are edited by hand often enough that a
missing bracket is a real failure mode: a syntax error in one file takes down
every agent in the folder at load time, not just the one being edited. Every
command here goes through the same AgentRegistry the pipeline loads, so a config
that passes `validate` is exactly a config transcribe_test.py can start with.
"""

import argparse
import contextlib
import io
import json
import os
import re

from agent_platform.orchestrator import AGENTS_DIR, TOOLS_FILE, AgentRegistry

DEFAULT_LLM_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")
DEFAULT_TTS_VOICE = os.getenv("PIPER_MODEL")

# The LiveKit AudioSource fixes its rate at construction, so this is the number
# every configured voice has to match (same constant transcribe_test.py builds
# its source with). A voice at any other rate crashes the turn mid-publish.
TRACK_SAMPLE_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))

# An id becomes a filename, a tools.json key ("handoff_to_<id>") and a tool name
# the model has to emit verbatim, so it is restricted to what all three tolerate.
AGENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

MODEL_COL = 22
VOICE_COL = 26
TOOLS_COL = 30
HANDOFF_COL = 18


# --- reading config ---------------------------------------------------------

def config_paths():
    paths = []
    if os.path.isdir(AGENTS_DIR):
        paths += [
            os.path.join(AGENTS_DIR, name)
            for name in sorted(os.listdir(AGENTS_DIR))
            if name.endswith(".json")
        ]
    paths.append(TOOLS_FILE)
    return paths


def scan_config_files():
    """Every config file parsed on its own, so a mistake is reported against the
    file that caused it.

    AgentRegistry._load() uses json.load() straight off the handle, so a syntax
    error surfaces as "Expecting value: line 1 column 1" with no indication of
    which of the files is at fault. Pointing this tool at that error is the main
    reason it exists, so the filename and position are worth the extra pass.
    """
    problems = []

    if not os.path.isdir(AGENTS_DIR):
        return [f"agent config folder is missing: {AGENTS_DIR}"]

    for path in config_paths():
        label = os.path.relpath(path)
        if not os.path.exists(path):
            problems.append(f"{label}: file does not exist")
            continue
        try:
            with open(path) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            problems.append(f"{label}: invalid JSON at line {e.lineno} column {e.colno}: {e.msg}")
            continue
        except OSError as e:
            problems.append(f"{label}: {e}")
            continue

        problems += structural_problems(label, config, is_agent=path != TOOLS_FILE)

    return problems


def structural_problems(label, config, is_agent):
    """Shape checks per file, kept here rather than in the registry so each one
    can be attributed to a filename instead of raised as one opaque error."""
    problems = []

    if not isinstance(config, dict):
        return [f"{label}: must be a JSON object, got {type(config).__name__}"]

    if is_agent:
        for key in ("id", "system_prompt"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                problems.append(f"{label}: '{key}' is required and must be a non-empty string")
    else:
        # A tool with no description renders as a blank line in the prompt, which
        # the registry also rejects — naming the tool here is the useful part.
        for name, info in config.items():
            if not isinstance(info, dict):
                problems.append(f"{label}: tool '{name}' must be an object")
            elif not info.get("description"):
                problems.append(f"{label}: tool '{name}' has no description")

    for key in ("tools", "handoffs", "rules"):
        if key in config and not isinstance(config[key], list):
            problems.append(f"{label}: '{key}' must be a list, got {type(config[key]).__name__}")

    return problems


@contextlib.contextmanager
def muted():
    """Silences the registry's own load banner.

    AgentRegistry._load() prints what it loaded, which belongs in the pipeline's
    startup log. Every command here prints its own counts, so the banner would
    only land in the middle of the output as noise.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_registry():
    """Returns (registry, problems) without ever raising on a bad config.

    AgentRegistry._load() ends in validate_configs(), which raises. That is
    right for the pipeline — a bad config should stop startup — but a config
    editor still has to list and repair that config, so the failure is captured
    and the partially loaded registry is handed back for the read paths.
    """
    registry = AgentRegistry.__new__(AgentRegistry)
    registry.default_llm_model = DEFAULT_LLM_MODEL
    registry.default_tts_voice = DEFAULT_TTS_VOICE
    registry.default_sample_rate = TRACK_SAMPLE_RATE
    registry.agents = {}
    registry.tools = {}
    registry._tool_args = {}

    problems = []
    try:
        with muted():
            registry._load()
    except ValueError as e:
        # validate_configs() and _parse_tool_args() both land here, and both
        # already format their own multi-line report.
        problems.append(str(e))
    except KeyError as e:
        problems.append(f"config is missing the required key {e}")
    except json.JSONDecodeError as e:
        problems.append(f"invalid JSON at line {e.lineno} column {e.colno}: {e.msg}")
    except OSError as e:
        problems.append(f"could not read config: {e}")

    return registry, problems


# --- prompting --------------------------------------------------------------

def ask(prompt, default=None):
    label = f"{prompt} [{default}]" if default else prompt
    while True:
        answer = input(f"  {label}: ").strip()
        if answer:
            return answer
        if default:
            return default
        print("    (required)")


def ask_multiline(prompt, default=None, hint=None, optional=False):
    if hint is None:
        hint = ("type across as many lines as you need, empty line to finish"
                if default is None else "empty line keeps it, '-' clears it")
    print(f"\n{prompt}")
    print(f"  ({hint})")
    lines = []
    while True:
        try:
            line = input("  > ")
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line.strip())

    # A lone '-' is how an existing multi-line value gets emptied: an empty line
    # on its own means "unchanged", which would leave no way to remove a rule.
    if lines == ["-"]:
        return ""
    if not lines:
        if default is not None:
            return default
        if optional:
            return None
        print("    (required)")
        return ask_multiline(prompt, default, hint, optional)
    return "\n".join(lines)


def ask_yes_no(question, default=True):
    answer = input(f"  {question} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def ask_multiselect(heading, options, defaults=()):
    """options is a list of (value, label) shown numbered. Accepts numbers or
    names, comma or space separated; an empty answer keeps the defaults, which
    is what makes re-running an edit a no-op instead of a reset."""
    if not options:
        print(f"\n{heading}\n  (nothing available)")
        return []

    defaults = set(defaults)
    print(f"\n{heading}")
    for i, (value, label) in enumerate(options, 1):
        mark = "x" if value in defaults else " "
        print(f"   {mark} [{i}] {value:<22} {label}")

    hint = f" (empty keeps: {', '.join(sorted(defaults))})" if defaults else ""
    while True:
        raw = input(f"   numbers or names, comma-separated{hint}: ").strip()
        if not raw:
            return [v for v, _ in options if v in defaults]

        chosen = []
        unknown = []
        for token in raw.replace(",", " ").split():
            match = next(
                (v for i, (v, _) in enumerate(options, 1)
                 if token == str(i) or token.lower() == v.lower()),
                None,
            )
            if match is None:
                unknown.append(token)
            elif match not in chosen:
                chosen.append(match)
        if unknown:
            print(f"    not on the list: {', '.join(unknown)}")
            continue
        return chosen


# --- writing config ---------------------------------------------------------

def write_json(path, data):
    """Written to a sibling temp file and renamed into place, because a
    half-written agent file breaks the whole registry rather than just the agent
    being edited."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def handoff_tool_entry(target):
    """The shape every handoff tool already in tools.json uses: a description for
    the prompt, is_handoff + target for resolve_handoff(), and one free-text
    reason argument."""
    return {
        "description": f"Transfers the conversation to {target}",
        "is_handoff": True,
        "target": target,
        "args_schema": {
            "reason": {
                "type": "string",
                "description": "Why the user is being transferred",
                "example": f"needs {target}",
            }
        },
    }


def ensure_handoff_tools(tools, handoffs):
    """Adds the handoff_to_<target> entry for any chosen handoff that lacks one.

    validate_configs() rejects an agent that hands off to a target with no
    matching tool, because build_prompt() looks the generated name up in
    self.tools. Creating an agent that would be refused on load is the one
    outcome this tool must not produce, so the entry is offered here rather than
    left for a hand edit of tools.json.
    """
    missing = [t for t in handoffs if f"handoff_to_{t}" not in tools]
    if not missing:
        return

    print()
    for target in missing:
        print(f"  tools.json has no 'handoff_to_{target}' entry.")
    print("  A handoff is rendered as a tool, and an agent naming a handoff with no\n"
          "  matching entry is rejected at load — so this agent would not start.")
    if not ask_yes_no("Add the standard entries now?", default=True):
        print("  Skipped. Run `validate` before starting the agent to see the gap.")
        return

    for target in missing:
        tools[f"handoff_to_{target}"] = handoff_tool_entry(target)
    write_json(TOOLS_FILE, tools)
    print(f"  Added {', '.join('handoff_to_' + t for t in missing)} to {os.path.relpath(TOOLS_FILE)}")


# --- agent fields -----------------------------------------------------------

def summarize(text, width=54):
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def edit_fields(agent, tools, other_agents):
    """Prompts for every field on `agent` in place, so create is just an empty
    dict and edit is the same flow seeded with what is already there.

    other_agents maps the ids this agent may hand off to, to their configs — the
    handoff list is the one place where picking the wrong name silently produces
    an agent that never transfers, so each option shows what that agent is.
    """
    agent["system_prompt"] = ask_multiline("System prompt:", agent.get("system_prompt"))

    # Handoff tools are generated from the handoffs list by build_prompt(), so
    # offering them here too would render the same tool twice in the prompt.
    tool_options = sorted(
        (name, summarize(info.get("description", ""), 48))
        for name, info in tools.items()
        if not info.get("is_handoff")
    )
    print("\n(tools.json handoff tools are generated from the handoffs list below.)")
    agent["tools"] = ask_multiselect("Tools this agent may call:", tool_options,
                                     agent.get("tools", []))

    handoff_options = sorted(
        (agent_id, summarize(config.get("system_prompt", "")))
        for agent_id, config in other_agents.items()
    )
    agent["handoffs"] = ask_multiselect("Handoffs (agents it can transfer to):",
                                        handoff_options, agent.get("handoffs", []))

    rules_text = ask_multiline(
        "Rules — uppercase RULE lines this small model must follow, one per line:",
        "\n".join(agent["rules"]) if agent.get("rules") else None,
        hint="one rule per line, empty line for none, '-' clears existing",
        optional=True,
    )
    agent["rules"] = [line for line in (rules_text or "").splitlines() if line.strip()]
    # Phi-4-mini ignores soft suggestions, which is the documented reason these
    # agents carry explicit RULEs at all — worth flagging when one is missing it.
    for rule in agent["rules"]:
        if "MUST" not in rule:
            print(f"    warning: '{summarize(rule, 50)}' has no MUST — a small local "
                  f"model reads it as a suggestion")

    model = ask("LLM model — empty uses the platform default",
                agent.get("llm_model") or DEFAULT_LLM_MODEL)
    agent["llm_model"] = None if model == DEFAULT_LLM_MODEL else model

    # The platform default is passed as the fallback rather than left blank, so
    # accepting it lands on the same "no override in the file" state as leaving
    # the field out would.
    default_voice = os.path.basename(DEFAULT_TTS_VOICE) if DEFAULT_TTS_VOICE else "(PIPER_MODEL)"
    voice = ask(f"Piper voice .onnx path — empty uses the platform default ({default_voice})",
                agent.get("tts_voice") or default_voice)
    if voice != default_voice and not os.path.exists(voice):
        print(f"    warning: {voice} does not exist — `validate` will fail on it")
    agent["tts_voice"] = None if voice == default_voice else voice

    for optional in ("llm_model", "tts_voice"):
        if agent[optional] is None:
            del agent[optional]
    return agent


def report_saved(agent_id, path):
    """Re-reads the file from disk through the registry, so "the orchestrator can
    load what I just wrote" is checked rather than assumed — the point of the
    whole tool."""
    print(f"\nWrote {os.path.relpath(path)}")

    registry, problems = load_registry()
    problems = scan_config_files() + problems
    if problems:
        for problem in problems:
            print(f"   [FAIL] {problem}")
        print("\nThe file is on disk but does not load cleanly — fix the lines above.")
        return False

    try:
        prompt = registry.build_prompt(agent_id, "Hello")
        providers = registry.resolve_providers(agent_id)
        registry.validate_voice_sample_rates(TRACK_SAMPLE_RATE)
    except (KeyError, ValueError) as e:
        print(f"   [FAIL] {e}")
        return False

    voice = os.path.basename(providers["tts_voice"])
    print(f"   [ok] loads, prompt renders ({len(prompt)} chars), "
          f"model={providers['llm_model']} voice={voice} @ {TRACK_SAMPLE_RATE} Hz")
    return True


def cmd_create(args):
    registry, problems = load_registry()
    for problem in problems:
        print(f"   [note] existing config problem: {summarize(problem, 100)}")

    while True:
        agent_id = ask("Agent id (lowercase, underscores — becomes handoff_to_<id>)")
        if not AGENT_ID_RE.match(agent_id):
            print("    must start with a letter and use only a-z, 0-9 and _")
            continue
        if agent_id in registry.agents:
            print(f"    '{agent_id}' already exists — use `edit-agent`")
            continue
        break

    path = os.path.join(AGENTS_DIR, f"{agent_id}.json")
    print(f"\nCreating {os.path.relpath(path)}. Enter to accept the value in [brackets].\n")

    # An agent cannot hand off to one that does not exist yet, so the only ids
    # on offer here are the ones already loaded.
    agent = edit_fields({"id": agent_id}, registry.tools, registry.agents)

    ensure_handoff_tools(registry.tools, agent["handoffs"])
    write_json(path, agent)
    return 0 if report_saved(agent_id, path) else 1


def cmd_edit(args):
    registry, problems = load_registry()
    if args.agent_id not in registry.agents:
        available = ", ".join(sorted(registry.agents)) or "none"
        print(f"   [FAIL] no agent '{args.agent_id}' ({available})")
        return 1
    for problem in problems:
        print(f"   [note] existing config problem: {summarize(problem, 100)}")

    agent = dict(registry.agents[args.agent_id])
    # The id is deliberately not re-prompted: it is the filename, and it is a key
    # in other agents' handoff_to_<id> tool entries, so renaming it here would
    # break every agent that hands off to this one.
    print(f"\nEditing {args.agent_id}. Enter to keep the value in [brackets].\n")

    # A handoff to itself is not something build_prompt() can act on.
    others = {a: c for a, c in registry.agents.items() if a != args.agent_id}
    edit_fields(agent, registry.tools, others)

    ensure_handoff_tools(registry.tools, agent["handoffs"])
    path = os.path.join(AGENTS_DIR, f"{args.agent_id}.json")
    write_json(path, agent)
    return 0 if report_saved(args.agent_id, path) else 1


# --- list -------------------------------------------------------------------

def wrap_items(items, width):
    """Comma-joined list folded at a column width. Kept instead of textwrap so an
    item is never split across lines — a tool name cut in half is unreadable."""
    lines, current = [], []
    for item in items:
        if current and len(", ".join(current + [item])) > width:
            lines.append(", ".join(current))
            current = [item]
        else:
            current.append(item)
    if current:
        lines.append(", ".join(current))
    return lines or ["—"]


def cmd_list(args):
    registry, problems = load_registry()
    if not registry.agents:
        print(f"No agents in {os.path.relpath(AGENTS_DIR)}/ — `create-agent` to add one.")
        return 1

    rows = []
    for agent_id, agent in sorted(registry.agents.items()):
        try:
            providers = registry.resolve_providers(agent_id)
        except ValueError:
            # Unresolvable only when neither the agent nor the platform has a
            # value; the config is still worth listing, just flagged.
            model, voice = "(unresolved)", "(unresolved)"
        else:
            model, voice = providers["llm_model"], os.path.basename(providers["tts_voice"])
        rows.append([
            [agent_id],
            [model[: MODEL_COL - 1] + "…" if len(model) > MODEL_COL else model],
            [voice[: VOICE_COL - 1] + "…" if len(voice) > VOICE_COL else voice],
            wrap_items(agent.get("tools", []), TOOLS_COL),
            wrap_items(agent.get("handoffs", []), HANDOFF_COL),
        ])

    headers = ["ID", "MODEL", "VOICE", "TOOLS", "HANDOFFS"]
    widths = [
        max(len(headers[i]), max(len(line) for line in column))
        for i, column in enumerate(zip(*rows))
    ]

    def line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        for i in range(max(len(column) for column in row)):
            print(line([column[i] if i < len(column) else "" for column in row]))
        print()

    print(f"{len(registry.agents)} agents, {len(registry.tools)} tools in "
          f"{os.path.relpath(TOOLS_FILE)}. '—' means none.")
    if problems:
        print(f"{len(problems)} config problem(s) — run `validate` for detail.")
    return 0


# --- validate ---------------------------------------------------------------

def cmd_validate(args):
    print(f"Validating {os.path.relpath(AGENTS_DIR)}/ and {os.path.relpath(TOOLS_FILE)}\n")

    registry, problems = load_registry()
    file_problems = scan_config_files()
    ok = True

    print("1. Config files parse and have the required fields")
    if file_problems:
        ok = False
        for problem in file_problems:
            print(f"   [FAIL] {problem}")
    else:
        print(f"   [ok] {len(registry.agents)} agents, {len(registry.tools)} tools")

    # A file that does not parse aborts AgentRegistry._load() partway, so
    # self.tools is never populated and every cross-reference below would be
    # reported as an unknown tool. Those are consequences of the one real
    # problem, so the dependent checks are skipped rather than adding noise
    # that sends the reader looking in the wrong file.
    dependent = not file_problems
    skip_note = "  [skip] a file above did not load, so this check would only " \
                "report consequences of it — fix step 1 and re-run"

    print("2. Tool schemas, tool references and handoff targets")
    if not dependent:
        print(skip_note)
    elif problems:
        ok = False
        for line in problems[0].splitlines():
            print(f"   {line}")
    else:
        print("   [ok] every listed tool exists and every handoff target resolves")

    print(f"3. Piper voice sample rates against the {TRACK_SAMPLE_RATE} Hz audio track")
    try:
        report = registry.validate_voice_sample_rates(TRACK_SAMPLE_RATE)
    except ValueError as e:
        ok = False
        for line in str(e).splitlines():
            print(f"   {line}")
    else:
        if not report:
            ok = False
            print("   [FAIL] no voices configured — set PIPER_MODEL in .env or give an "
                  "agent a tts_voice")
        for row in report:
            flag = "ok" if row["matches_track"] else "FAIL"
            print(f"   [{flag}] [{row['source']}] {row['voice']} @ {row['sample_rate']} Hz")

    print("4. Per-agent provider resolution and prompt build")
    if not dependent:
        print(skip_note)
    elif not registry.agents:
        ok = False
        print("   [FAIL] no agents loaded")
    else:
        for agent_id in sorted(registry.agents):
            try:
                providers = registry.resolve_providers(agent_id)
            except ValueError as e:
                ok = False
                print(f"   [FAIL] {e}")
                continue
            # build_prompt() is the check that matters most here: it indexes
            # self.tools by every declared name, so it is where a bad tool or
            # handoff reference would first actually fail.
            try:
                prompt = registry.build_prompt(agent_id, "Hello")
            except (KeyError, ValueError) as e:
                ok = False
                print(f"   [FAIL] {agent_id}: {e}")
                continue
            overrides = [
                label for label, key in (("own model", "llm_model_overridden"),
                                         ("own voice", "tts_voice_overridden"))
                if providers[key]
            ]
            note = f" ({', '.join(overrides)})" if overrides else ""
            print(f"   [ok] {agent_id:<12} {providers['llm_model']:<16} "
                  f"{os.path.basename(providers['tts_voice']):<24} "
                  f"prompt {len(prompt):>4} chars{note}")

    print()
    if ok:
        print("All checks passed — transcribe_test.py can start with this config.")
        return 0
    print("Config has problems — fix the [FAIL] lines above, then re-run validate.")
    return 1


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create and edit agent JSON configs without hand-writing them.")
    subs = parser.add_subparsers(dest="command", required=True)

    create = subs.add_parser("create-agent", help="create a new agent config")
    create.set_defaults(func=cmd_create)

    edit = subs.add_parser("edit-agent", help="edit an existing agent config")
    edit.add_argument("agent_id", help="id of the agent to edit")
    edit.set_defaults(func=cmd_edit)

    listing = subs.add_parser("list-agents", help="list agents with tools and handoffs")
    listing.set_defaults(func=cmd_list)

    validate = subs.add_parser("validate", help="check the config without starting the pipeline")
    validate.set_defaults(func=cmd_validate)

    return parser


def main():
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
