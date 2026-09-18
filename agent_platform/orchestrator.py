import json
import os

AGENTS_DIR = os.path.join(os.path.dirname(__file__), "agents")
TOOLS_FILE = os.path.join(os.path.dirname(__file__), "tools.json")


class AgentRegistry:
    """Loads agent configs and the tool registry, and builds prompts generically —
    no hardcoded per-agent branching anywhere in this class."""

    def __init__(self):
        self.agents = {}
        self.tools = {}
        self._load()

    def _load(self):
        # Load every agent config in the agents/ folder
        for filename in os.listdir(AGENTS_DIR):
            if filename.endswith(".json"):
                with open(os.path.join(AGENTS_DIR, filename)) as f:
                    agent = json.load(f)
                    self.agents[agent["id"]] = agent

        # Load the tool registry
        with open(TOOLS_FILE) as f:
            self.tools = json.load(f)

        print(f"[ORCHESTRATOR] Loaded {len(self.agents)} agents: {list(self.agents.keys())}")
        print(f"[ORCHESTRATOR] Loaded {len(self.tools)} tools: {list(self.tools.keys())}")

    def get_agent(self, agent_id):
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        return self.agents[agent_id]

    def build_prompt(self, agent_id, user_text, context_note=""):
        """Generically builds a tool-calling prompt for ANY agent, based purely
        on its config — this replaces the old hardcoded if/else branch."""
        agent = self.get_agent(agent_id)

        # Collect this agent's own tools + auto-generated handoff tools
        tool_lines = []
        json_examples = []

        for tool_name in agent.get("tools", []):
            tool_info = self.tools.get(tool_name)
            if not tool_info:
                continue
            tool_lines.append(f"- {tool_info['description']}")
            args_placeholder = self._infer_args_placeholder(tool_name)
            json_examples.append(f'{{"tool": "{tool_name}", "args": {args_placeholder}}}')

        for target_agent_id in agent.get("handoffs", []):
            handoff_tool_name = f"handoff_to_{target_agent_id}"
            tool_info = self.tools.get(handoff_tool_name)
            if tool_info:
                tool_lines.append(f"- {tool_info['description']}")
                json_examples.append(f'{{"tool": "{handoff_tool_name}", "args": {{"reason": "..."}}}}')

        rules_block = ""
        if agent.get("rules"):
            rules_block = "\n".join(f"RULE: {r}" for r in agent["rules"]) + "\n"

        prompt = f"""{agent['system_prompt']}{context_note}

You have access to these tools:
{chr(10).join(tool_lines)}

{rules_block}If the user's request needs one of these, respond with ONLY the matching JSON and nothing else:
{chr(10).join(json_examples)}

Otherwise, respond normally and conversationally. Do NOT explain your reasoning or mention tools.

User: {user_text}
Agent:"""
        return prompt

    def _infer_args_placeholder(self, tool_name):
        # Small helper so we don't need to hardcode arg shapes per tool in two places.
        # For a more complex platform, tools.json would carry a real JSON schema per tool
        # instead of this inference — noted as a known simplification.
        if tool_name == "calculate":
            return '{"expression": "..."}'
        if tool_name == "check_calendar":
            return '{"date": "..."}'
        return "{}"

    def resolve_handoff(self, tool_name):
        """Given a tool name, returns the target agent id if this tool is a handoff, else None."""
        tool_info = self.tools.get(tool_name)
        if tool_info and tool_info.get("is_handoff"):
            return tool_info.get("target")
        return None


if __name__ == "__main__":
    # Standalone test — no LiveKit, no live pipeline, just confirm config loading + prompt building works.
    registry = AgentRegistry()

    print("\n--- Testing primary agent prompt ---")
    print(registry.build_prompt("primary", "What's 12 plus 15?"))

    print("\n--- Testing scheduler agent prompt ---")
    print(registry.build_prompt("scheduler", "Do I have anything on August 16th?", context_note="\n(Context from handoff: user wants to find a free slot)"))

    print("\n--- Testing handoff resolution ---")
    print("handoff_to_scheduler resolves to:", registry.resolve_handoff("handoff_to_scheduler"))
    print("calculate resolves to:", registry.resolve_handoff("calculate"))