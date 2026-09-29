import copy
import json
import os

from dotenv import load_dotenv

# The standalone self-test below reads PIPER_MODEL, and the library stays free of
# environment lookups by design, so the .env is loaded here rather than in the
# class. Without it the self-test dies on "no tts_voice and no platform default".
load_dotenv()

AGENTS_DIR = os.path.join(os.path.dirname(__file__), "agents")
TOOLS_FILE = os.path.join(os.path.dirname(__file__), "tools.json")

# JSON Schema types a tool argument may declare. Anything else is a config typo,
# and it is rejected when the registry loads rather than silently reaching the
# prompt, where it would show up as a wrong example value mid-call.
ARG_TYPES = ("string", "number", "integer", "boolean", "array", "object", "null")

# Value emitted for an argument whose schema declares no "example". Only strings
# need prose here — for the rest, the type's zero value is self-explanatory.
TYPE_ZERO_VALUES = {
    "number": 0,
    "integer": 0,
    "boolean": False,
    "array": [],
    "object": {},
    "null": None,
}


class AgentRegistry:
    """Loads agent configs and the tool registry, and builds prompts generically —
    no hardcoded per-agent branching anywhere in this class."""

    def __init__(self, default_llm_model=None, default_tts_voice=None,
                 default_sample_rate=None, config_store=None):
        self.agents = {}
        self.tools = {}
        # When set, config comes from SQLite and is re-read whenever the store's
        # config_version changes. When None, the JSON files remain the source of
        # truth, which is what keeps this class usable with no database present.
        self.config_store = config_store
        self._loaded_config_version = None
        # Argument descriptors per tool, parsed from tools.json once at load so a
        # malformed schema fails at startup and build_prompt() stays a pure
        # formatter with nothing to parse per turn.
        self._tool_args = {}
        # Platform defaults are injected rather than read from the environment here,
        # so this class stays a pure config layer and stays unit-testable offline.
        self.default_llm_model = default_llm_model
        self.default_tts_voice = default_tts_voice
        self.default_sample_rate = default_sample_rate
        self._load()

    def _load(self):
        if self.config_store is not None:
            snapshot = self.config_store.load_snapshot()
            self.agents = snapshot["agents"]
            self.tools = snapshot["tools"]
            self._loaded_config_version = self.config_store.config_version()
        else:
            # Load every agent config in the agents/ folder
            for filename in os.listdir(AGENTS_DIR):
                if filename.endswith(".json"):
                    with open(os.path.join(AGENTS_DIR, filename)) as f:
                        agent = json.load(f)
                        self.agents[agent["id"]] = agent

            # Load the tool registry
            with open(TOOLS_FILE) as f:
                self.tools = json.load(f)

        self._parse_tool_args()
        self.validate_configs()

    def refresh_if_changed(self):
        """Re-read config if the store's version has moved. Returns True if it did.

        Called once per new session, never mid-call. A mid-call reload would swap
        the tool list underneath a prompt that was already built from the old one,
        so a tool call the model just made could be rejected as unknown — a
        failure that looks like a model bug and is really a config race.
        """
        if self.config_store is None:
            return False
        current = self.config_store.config_version()
        if current == self._loaded_config_version:
            return False
        self.agents = {}
        self.tools = {}
        self._load()
        print(
            f"[ORCHESTRATOR] Config changed (v{self._loaded_config_version}); "
            f"reloaded {len(self.agents)} agents"
        )
        return True

        print(f"[ORCHESTRATOR] Loaded {len(self.agents)} agents: {list(self.agents.keys())}")
        print(f"[ORCHESTRATOR] Loaded {len(self.tools)} tools: {list(self.tools.keys())}")

    def _parse_tool_args(self):
        """Turn every tool's args_schema into normalized argument descriptors.

        A schema is either the flat form the config uses —
        {"expression": {"type": "string", "description": "..."}} — or a standard
        JSON Schema object — {"type": "object", "properties": {...},
        "required": [...]} — so a schema can be written or pasted in either shape.
        In the flat form an argument is required unless it says otherwise; in the
        JSON Schema form "required" decides, and falls back to the same default
        when the key is absent.
        """
        self._tool_args = {}
        for tool_name, tool_info in self.tools.items():
            if not isinstance(tool_info, dict):
                raise ValueError(f"tool '{tool_name}' must be an object")

            schema = tool_info.get("args_schema")
            if schema is None:
                self._tool_args[tool_name] = []
                continue
            if not isinstance(schema, dict):
                raise ValueError(
                    f"tool '{tool_name}': args_schema must be an object, "
                    f"got {type(schema).__name__}"
                )

            if "properties" in schema:
                properties = schema["properties"]
                if not isinstance(properties, dict):
                    raise ValueError(f"tool '{tool_name}': args_schema.properties must be an object")
                required_names = schema.get("required")
                if required_names is not None and not isinstance(required_names, list):
                    raise ValueError(f"tool '{tool_name}': args_schema.required must be a list")
            else:
                properties = schema
                required_names = None

            args = []
            for arg_name, spec in properties.items():
                where = f"tool '{tool_name}' arg '{arg_name}'"
                if not isinstance(spec, dict):
                    raise ValueError(
                        f"{where}: each argument must be an object with 'type' and 'description', "
                        f"got {type(spec).__name__}"
                    )

                arg_type = spec.get("type", "string")
                if arg_type not in ARG_TYPES:
                    raise ValueError(
                        f"{where}: unknown type '{arg_type}' "
                        f"(expected one of {', '.join(ARG_TYPES)})"
                    )

                description = spec.get("description")
                if description is not None and not isinstance(description, str):
                    raise ValueError(f"{where}: description must be a string")

                enum = spec.get("enum")
                if enum is not None and (not isinstance(enum, list) or not enum):
                    raise ValueError(f"{where}: enum must be a non-empty list")

                if required_names is not None:
                    required = arg_name in required_names
                else:
                    required = bool(spec.get("required", True))

                args.append({
                    "name": arg_name,
                    "type": arg_type,
                    "description": description or "",
                    "required": required,
                    "example": spec.get(
                        "example", self._placeholder_example(arg_type, description, enum)
                    ),
                })

            self._tool_args[tool_name] = args

    @staticmethod
    def _placeholder_example(arg_type, description, enum):
        """The value shown to the LLM for an argument with no explicit "example".

        A string falls back to its own description, because for a 3B-class model
        that text is the only signal about what belongs in the slot — and it is
        worth more than a bare "...", which is what the old hardcoded inference
        emitted. The leading article is dropped so the text reads as a value
        rather than a sentence. Other types fall back to their zero value, and a
        pinned enum uses its first legal value.
        """
        if enum:
            return enum[0]
        if arg_type != "string":
            # Copied, because a list/dict zero value would otherwise be shared by
            # every tool that declares an array or object argument.
            return copy.copy(TYPE_ZERO_VALUES[arg_type])
        if not description:
            return "..."
        text = description.strip().rstrip(".")
        for prefix in ("The ", "A ", "An "):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        return text[:1].lower() + text[1:] if text else "..."

    def validate_configs(self):
        """Fails at startup on a config that would otherwise misbehave mid-call.

        Every tool is now defined entirely in tools.json, so both ways a
        config-only change can go wrong are caught here: a tool whose schema does
        not parse (raised by _parse_tool_args) and an agent naming a tool that
        does not exist. The old code skipped an unknown tool silently, so a typo
        in an agent's tools list quietly left that agent with fewer tools than
        its config claimed.
        """
        problems = []

        for tool_name, tool_info in self.tools.items():
            if not tool_info.get("description"):
                problems.append(f"tool '{tool_name}' has no description")
            if tool_info.get("is_handoff") and tool_info.get("target") not in self.agents:
                problems.append(
                    f"handoff tool '{tool_name}' targets unknown agent '{tool_info.get('target')}'"
                )

        for agent_id, agent in self.agents.items():
            for tool_name in agent.get("tools", []):
                if tool_name not in self.tools:
                    problems.append(f"agent '{agent_id}' lists unknown tool '{tool_name}'")
            for target_id in agent.get("handoffs", []):
                if target_id not in self.agents:
                    problems.append(f"agent '{agent_id}' hands off to unknown agent '{target_id}'")
                elif f"handoff_to_{target_id}" not in self.tools:
                    problems.append(
                        f"agent '{agent_id}' hands off to '{target_id}' but tools.json has no "
                        f"'handoff_to_{target_id}' entry"
                    )

        if problems:
            raise ValueError("Invalid agent/tool config:\n  " + "\n  ".join(problems))

    def tool_args(self, tool_name):
        """Normalized argument descriptors for a tool, as declared in tools.json.

        Each is {name, type, description, required, example}. This is the single
        source of truth for what a tool takes: build_prompt() renders it and
        normalize_args() enforces it, so neither has to know any tool by name.
        """
        return self._tool_args.get(tool_name, [])

    def get_agent(self, agent_id):
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")
        return self.agents[agent_id]

    def _render_tool_doc(self, tool_name, tool_info):
        """The human-readable tool line: name, generated signature, description,
        then one indented line per argument. Nothing here is tool-specific."""
        args = self.tool_args(tool_name)
        signature = ", ".join(
            f"{a['name']}: {a['type']}" + ("" if a["required"] else " (optional)")
            for a in args
        )
        header = f"- {tool_name}({signature})" if signature else f"- {tool_name}"
        lines = [f"{header}: {tool_info['description']}"]
        for a in args:
            if a["description"]:
                suffix = "" if a["required"] else ", optional"
                lines.append(f"    {a['name']}: {a['description']}{suffix}")
        return "\n".join(lines)

    def _render_tool_call(self, tool_name):
        """The JSON example, generated straight from the schema: the argument names
        as keys, and each argument's example (explicit or placeholder) as the
        value. The shape matches what get_llm_response() parses, so the example the
        model is shown is literally the format it must answer in."""
        payload = {a["name"]: a["example"] for a in self.tool_args(tool_name)}
        return json.dumps({"tool": tool_name, "args": payload}, ensure_ascii=False)

    def build_prompt(self, agent_id, user_text, context_note=""):
        """Generically builds a tool-calling prompt for ANY agent, based purely
        on its config — this replaces the old hardcoded if/else branch."""
        agent = self.get_agent(agent_id)

        # This agent's own tools plus the auto-generated handoff tools, both
        # rendered from the same schema-driven formatters, so a handoff is just
        # another declared tool rather than a separate code path.
        tool_names = list(agent.get("tools", [])) + [
            f"handoff_to_{target_id}" for target_id in agent.get("handoffs", [])
        ]

        tool_lines = []
        json_examples = []
        for tool_name in tool_names:
            tool_info = self.tools[tool_name]
            tool_lines.append(self._render_tool_doc(tool_name, tool_info))
            json_examples.append(self._render_tool_call(tool_name))

        rules_block = ""
        if agent.get("rules"):
            rules_block = "\n".join(f"RULE: {r}" for r in agent["rules"]) + "\n"

        # Scope rule, generated for any agent that can hand off. Without it a small
        # model treats "respond conversationally" as license to invent an answer for
        # anything outside its tools. The second RULE is the guard against the
        # opposite failure: without it the model hands off greetings and small talk
        # instead of just replying.
        if agent.get("handoffs") and os.getenv("SCOPE_RULE", "0") == "1":
            targets = ", ".join(agent["handoffs"])
            rules_block += (
                f"RULE: If the user's request needs a capability you do not have "
                f"(for example another agent's tools, or a subject outside your role), "
                f"you MUST hand off with one of these tools: {targets}. "
                f"Never invent or guess an answer to something outside your role.\n"
                f"RULE: Do NOT hand off for greetings, small talk, thanks, or any "
                f"question you can answer within your own role. Answer those yourself.\n"
            )

        prompt = f"""{agent['system_prompt']}{context_note}

You have access to these tools:
{chr(10).join(tool_lines)}

{rules_block}If the user's request needs one of these, respond with ONLY the matching JSON and nothing else:
{chr(10).join(json_examples)}

Otherwise, respond normally and conversationally. Do NOT explain your reasoning or mention tools.

User: {user_text}
Agent:"""
        return prompt

    def normalize_args(self, tool_name, args):
        """Coerces a parsed tool call's arguments to the types the schema declares.

        A small local model asked for a number usually sends "3" as a string, and
        it sometimes invents extra keys; passing either straight to MCP makes the
        call fail. Returns (args, notes) where notes records every correction, so
        the turn can log what the model got wrong instead of quietly hiding it.
        """
        declared = {a["name"]: a for a in self.tool_args(tool_name)}
        if not declared:
            return dict(args or {}), []

        clean = {}
        notes = []
        for name, value in (args or {}).items():
            if name not in declared:
                notes.append(f"dropped undeclared arg '{name}'")
                continue
            clean[name], note = self._coerce_arg(declared[name], value)
            if note:
                notes.append(f"'{name}' {note}")

        missing = sorted(n for n, a in declared.items() if a["required"] and n not in clean)
        if missing:
            notes.append("missing required arg(s): " + ", ".join(missing))

        return clean, notes

    @staticmethod
    def _coerce_arg(arg, value):
        """Best-effort cast of one value to its declared type. Uncastable values
        are passed through with a note rather than dropped — the MCP server is the
        real authority on the call, so this only fixes what it can fix."""
        arg_type = arg["type"]
        notes = []

        if arg_type in ("number", "integer"):
            if isinstance(value, bool):
                # bool is a subclass of int, so this has to be excluded before the
                # number branch or True would silently become 1.
                return value, "was a boolean, which is not a number, sent unchanged"
            number = value
            if isinstance(value, str):
                try:
                    number = float(value)
                except ValueError:
                    return value, f"is not a number ('{value}'), sent unchanged"
                notes.append(f"was the string '{value}'")
            if isinstance(number, float) and number.is_integer():
                number = int(number) if arg_type == "integer" else number
            if arg_type == "integer" and isinstance(number, float):
                number = int(number)
                notes.append("was rounded to an integer")
            return number, " and ".join(notes)

        if arg_type == "boolean" and not isinstance(value, bool):
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                notes.append(f"was the string '{value}'")
                return value.strip().lower() == "true", " and ".join(notes)
            return value, f"is not a boolean, sent unchanged"

        if arg_type == "string" and not isinstance(value, str):
            return str(value), f"was {type(value).__name__}, sent as a string"

        return value, None

    def resolve_handoff(self, tool_name):
        """Given a tool name, returns the target agent id if this tool is a handoff, else None."""
        tool_info = self.tools.get(tool_name)
        if tool_info and tool_info.get("is_handoff"):
            return tool_info.get("target")
        return None

    def resolve_providers(self, agent_id):
        """Returns the effective LLM model and TTS voice for an agent.

        Per-agent "llm_model" / "tts_voice" are optional; when absent the platform
        default is used, so adding an agent to the folder never requires touching
        the pipeline. The *_overridden flags let callers log which path was taken.
        """
        agent = self.get_agent(agent_id)

        llm_override = agent.get("llm_model")
        voice_override = agent.get("tts_voice")

        llm_model = llm_override or self.default_llm_model

        # A bare filename in tts_voice resolves beside the platform voice, so one
        # agent JSON works on a host and in a container without carrying a path
        # that only exists on one of them. An override that already has a
        # directory is taken as-is — that is the escape hatch for a voice kept
        # somewhere else entirely, and it stays the agent's own decision.
        tts_voice = self.resolve_voice_path(voice_override) or self.default_tts_voice

        if not llm_model:
            raise ValueError(
                f"Agent '{agent_id}' has no llm_model and no platform default was provided"
            )
        if not tts_voice:
            raise ValueError(
                f"Agent '{agent_id}' has no tts_voice and no platform default was provided"
            )

        return {
            "agent": agent_id,
            "llm_model": llm_model,
            "tts_voice": tts_voice,
            "llm_model_overridden": bool(llm_override),
            "tts_voice_overridden": bool(voice_override),
        }

    def resolve_voice_path(self, voice):
        """A bare tts_voice filename means "beside the platform voice".

        Shared by resolve_providers and all_configured_voices, because the
        startup check has to resolve the same way the turn does: if the check
        treated "en_US-amy-medium.onnx" as a relative path from the process CWD
        while the turn resolved it against PIPER_MODEL, startup would report a
        missing file for a voice that was about to work.
        """
        if voice and os.sep not in voice and self.default_tts_voice:
            return os.path.join(
                os.path.dirname(os.path.abspath(self.default_tts_voice)),
                voice,
            )
        return voice

    def all_configured_voices(self):
        """Every distinct voice that could ever be published into the live audio track:
        the platform default plus each per-agent override, deduped by path."""
        entries = []
        if self.default_tts_voice:
            entries.append(("platform_default", self.default_tts_voice))
        for agent_id in sorted(self.agents):
            voice = self.agents[agent_id].get("tts_voice")
            if voice:
                entries.append((agent_id, self.resolve_voice_path(voice)))

        by_path = {}
        for source, path in entries:
            by_path.setdefault(path, source)
        return [(source, path) for path, source in by_path.items()]

    @staticmethod
    def read_voice_sample_rate(voice_path):
        """Piper keeps a voice's native rate in the sibling <voice>.onnx.json config,
        so we can check it without synthesizing audio or adding onnxruntime."""
        config_path = voice_path + ".json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"no Piper config beside the voice (expected {config_path}); "
                "download the .onnx.json that ships with the voice"
            )
        with open(config_path) as f:
            config = json.load(f)
        rate = config.get("audio", {}).get("sample_rate")
        if not rate:
            raise ValueError(f"{config_path} has no audio.sample_rate")
        return int(rate)

    def validate_voice_sample_rates(self, track_sample_rate):
        """Startup guard for the audio-track gotcha: a LiveKit AudioSource's rate is
        fixed at construction, so publishing a frame at any other rate raises
        'InvalidState - sample_rate and num_channels don't match' mid-call.
        Checking every configured voice up front turns a runtime crash into a
        clear startup error naming the offending voice."""
        if not track_sample_rate:
            raise ValueError("validate_voice_sample_rates() requires the track's sample rate")

        report = []
        problems = []

        for source, path in self.all_configured_voices():
            if not os.path.exists(path):
                problems.append(f"  [{source}] voice file does not exist: {path}")
                continue
            try:
                rate = self.read_voice_sample_rate(path)
            except (OSError, ValueError, json.JSONDecodeError) as e:
                problems.append(f"  [{source}] could not read sample rate from {path}: {e}")
                continue

            ok = rate == track_sample_rate
            report.append({
                "source": source,
                "voice": os.path.basename(path),
                "sample_rate": rate,
                "matches_track": ok,
            })
            if not ok:
                problems.append(
                    f"  [{source}] {os.path.basename(path)} is {rate} Hz but the audio "
                    f"track is {track_sample_rate} Hz"
                )

        if problems:
            raise ValueError(
                "Piper voice sample-rate mismatch against the live audio track:\n"
                + "\n".join(problems)
                + "\nFix by pointing the agent at a voice whose native rate matches the "
                  "track, or by changing PIPER_SAMPLE_RATE to the rate all voices share."
            )

        return report


if __name__ == "__main__":
    # Standalone test — no LiveKit, no live pipeline, just confirm config loading + prompt building works.
    registry = AgentRegistry(
        default_llm_model=os.getenv("OLLAMA_MODEL", "phi4-mini"),
        default_tts_voice=os.getenv("PIPER_MODEL"),
    )

    print("\n--- Testing primary agent prompt ---")
    print(registry.build_prompt("primary", "What's 12 plus 15?"))

    print("\n--- Testing scheduler agent prompt ---")
    print(registry.build_prompt("scheduler", "Do I have anything on August 16th?", context_note="\n(Context from handoff: user wants to find a free slot)"))

    print("\n--- Testing handoff resolution ---")
    print("handoff_to_scheduler resolves to:", registry.resolve_handoff("handoff_to_scheduler"))
    print("calculate resolves to:", registry.resolve_handoff("calculate"))

    print("\n--- Testing schema-driven tool rendering (no tool name is special-cased) ---")
    for tool_name in sorted(registry.tools):
        args = registry.tool_args(tool_name)
        rendered = registry._render_tool_call(tool_name)
        # The example must be valid JSON in exactly the shape the parser expects.
        parsed = json.loads(rendered)
        assert parsed["tool"] == tool_name
        assert set(parsed["args"]) == {a["name"] for a in args}
        print(f"  {tool_name:22s} args={[(a['name'], a['type'], a['required']) for a in args]}")
        print(f"  {'':22s} {rendered}")

    print("\n--- Testing arg normalization against the declared schema ---")
    for tool_name in ("calculate", "check_calendar", "get_weather"):
        raw = {a["name"]: a["example"] for a in registry.tool_args(tool_name)}
        clean, notes = registry.normalize_args(tool_name, raw)
        print(f"  {tool_name}({raw}) -> {clean}  notes={notes}")

    # A number sent as a string, an invented key, and a missing required arg are the
    # three ways a small model gets a tool call wrong.
    print("\n--- Testing coercion / rejection of a bad call ---")
    print("  (against a throwaway schema, so no config is involved)")

    class _Demo(AgentRegistry):
        def _load(self):
            self.agents = {"demo": {"id": "demo", "system_prompt": "x", "tools": ["forecast"], "handoffs": []}}
            self.tools = {"forecast": {
                "description": "Gets a forecast",
                "args_schema": {"days": {"type": "integer", "description": "How many days ahead to look"},
                                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            }}
            self._parse_tool_args()
            self.validate_configs()

    demo_registry = _Demo()
    print("  example in prompt:", demo_registry._render_tool_call("forecast"))
    for raw in ({"days": "3", "unit": "celsius", "bogus": 1}, {"days": 2.0}, {"unit": "celsius"}, {}):
        clean, notes = demo_registry.normalize_args("forecast", raw)
        print(f"  {raw} -> {clean}  notes={notes}")

    print("\n--- Testing a bad schema is rejected at load ---")
    for bad in ({"type": "sting"}, {"x": {"type": "str"}}, {"x": "just a string"}):
        broken = AgentRegistry.__new__(AgentRegistry)
        broken._tool_args = {}
        broken.tools = {"broken": {"description": "d", "args_schema": bad}}
        try:
            broken._parse_tool_args()
            print(f"  {bad} -> NOT REJECTED (bug)")
        except ValueError as e:
            print(f"  {bad} -> rejected: {e}")

    print("\n--- Testing provider resolution per agent ---")
    for agent_id in sorted(registry.agents):
        p = registry.resolve_providers(agent_id)
        print(
            f"  {agent_id:10s} model={p['llm_model']:<10s} voice={os.path.basename(p['tts_voice'])}"
            f"  (model_override={p['llm_model_overridden']}, voice_override={p['tts_voice_overridden']})"
        )

    print("\n--- Testing voice sample-rate validation ---")
    try:
        for row in registry.validate_voice_sample_rates(22050):
            flag = "ok" if row["matches_track"] else "MISMATCH"
            print(f"  [{row['source']}] {row['voice']} @ {row['sample_rate']} Hz -> {flag}")
    except ValueError as e:
        print(e)
