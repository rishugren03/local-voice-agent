"""Checks the graceful-degradation paths without LiveKit, Ollama or Piper.

Each stage that can die mid-conversation (whisper.cpp, Ollama, MCP, Piper) is
exercised against a deliberately broken stand-in, and the assertion is always the
same: the turn produces a spoken answer and a trace event naming the cause,
rather than an exception that unwinds the turn task and leaves the caller
hearing nothing.

Run: python3 test_degradation.py
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# The import runs validate_config(), so the paths have to look real to it.
os.environ.setdefault("WHISPER_BIN", "/bin/true")
os.environ.setdefault("WHISPER_MODEL", "/bin/true")
os.environ.setdefault("PIPER_MODEL", "/bin/true")
# A port nothing is listening on, so the LLM calls below fail for the same
# reason they would if Ollama had been killed mid-conversation.
os.environ["OLLAMA_URL"] = "http://127.0.0.1:9"
os.environ["LLM_RETRY_BACKOFF_S"] = "0.05"
os.environ["MCP_TOOL_TIMEOUT_S"] = "0.2"
os.environ["TRACE_DB"] = os.path.join(tempfile.mkdtemp(), "degradation.db")

import transcribe_test as agent
from transcribe_test import LLMUnavailable

PASS, FAIL = [], []

# Captured before any test replaces it, so the Piper and STT sections below can
# exercise the real functions.
REAL_SPEAK = agent.speak
REAL_CALL_OLLAMA_ASYNC = agent.call_ollama_async


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


class FakeSession:
    """Enough of a Session for the stage functions under test."""

    def __init__(self):
        self.id = "degradation-test"
        self.active = "primary"
        self.context = ""
        self.turns_served = 0
        self.interrupts = 0
        self.playback = agent.PlaybackState()
        self.spoken = []
        self.source = None
        self.track = None
        self.publication = None
        self.mcp = None
        self.mcp_task = None
        self.turn_lock = asyncio.Lock()

    def __getattr__(self, name):
        # The stage functions poke at Session attributes these paths never reach.
        raise AttributeError(name)


def spoken_texts(session):
    """The text of every line the fake speak() was asked to say."""
    return [entry[1] for entry in session.spoken]


async def _record_speak(session, text, call_id, providers=None):
    """Stands in for the real Piper call, recording what would have been said."""
    session.spoken.append((session, text, call_id))


def main():
    print("Graceful degradation checks")
    # transcribe_test only builds the real registry inside main(), which needs
    # LiveKit; a stub is enough for the stage functions under test.
    agent.REGISTRY = _RegistryWithTool()

    # --- 1. call_ollama retries once, then gives up with a typed error ------
    print("\n1. call_ollama against an unreachable Ollama")
    t0 = time.monotonic()
    try:
        agent.call_ollama("hello", max_tokens=5)
        raised = None
    except LLMUnavailable as e:
        raised = e
    elapsed = time.monotonic() - t0
    check("raises LLMUnavailable instead of a requests exception",
          isinstance(raised, LLMUnavailable), f"{type(raised).__name__}: {raised}")
    check("the cause is named in the message",
          raised is not None and "Error" in str(raised), str(raised))
    check("it waited for the retry's backoff rather than giving up at once",
          elapsed >= agent.LLM_RETRY_BACKOFF_S, f"{elapsed:.2f}s")

    # --- 2. a whole turn degrades to the spoken fallback ---------------------
    print("\n2. A turn whose LLM is down still answers, out loud")
    session = FakeSession()
    agent.speak = _record_speak
    asyncio.run(agent.get_llm_response(session, "what is 12 plus 15?", "c1"))
    texts = spoken_texts(session)
    check("the caller still hears something",
          len(texts) == 1 and texts[0] == agent.LLM_FALLBACK_TEXT,
          f"said: {texts}")
    check("the fallback names the problem the caller hit",
          "trouble" in (texts[0].lower() if texts else ""))

    # --- 3. the trace records why -------------------------------------------
    print("\n3. The trace explains the degradation")
    from agent_platform.trace_store import connect_readonly
    conn = connect_readonly(os.environ["TRACE_DB"])
    events = [r["event_type"] for r in conn.execute(
        "SELECT event_type FROM events WHERE session_id = ? ORDER BY ts", (session.id,))]
    failed = [r["error"] for r in conn.execute(
        "SELECT error FROM events WHERE event_type = 'llm_unavailable'")]
    conn.close()
    check("an llm_unavailable event was written", "llm_unavailable" in events, f"{events}")
    check("it carries the underlying error", bool(failed and failed[0]), failed[0] if failed else "")

    # --- 4. a failing MCP tool degrades the turn -----------------------------
    print("\n4. A tool that fails or times out")
    # The tool path is only reached if the LLM asks for the tool in the first
    # place, so the model is stubbed to emit that call. Ollama is deliberately
    # still down here — what is under test is the tool's failure, and the LLM's
    # own degradation is already covered above.
    async def _llm_emits_tool_call(prompt, stop=None, max_tokens=150, model=None):
        if "You used the tool" in prompt:
            return "Here is what the tool said."
        return '{"tool": "always_fails", "args": {}}'

    agent.call_ollama_async = _llm_emits_tool_call
    session = FakeSession()
    session.spoken.clear()
    session.mcp = _McpStub(mode="raise")
    asyncio.run(agent._run_llm_turn(session, "what is 12 plus 15?", "c2"))
    texts = spoken_texts(session)
    check("the caller is told the action did not complete",
          len(texts) == 1 and texts[0] == agent.TOOL_FAILURE_TEXT, f"said: {texts}")

    session.spoken.clear()
    session.mcp = _McpStub(mode="hang")
    asyncio.run(agent._run_llm_turn(session, "what is 12 plus 15?", "c3"))
    texts = spoken_texts(session)
    check("a tool that never returns is cut off, not waited on forever",
          len(texts) == 1 and texts[0] == agent.TOOL_FAILURE_TEXT, f"said: {texts}")

    # A client that never started is the case that produced
    # "AttributeError: 'NoneType' object has no attribute 'call_tool'" in the
    # container: the stdio subprocess died on startup (wrong interpreter, no
    # mcp package) and the turn then called through None. The turn must still
    # degrade, and the trace must name the real cause rather than a Python
    # internal — the message is the only clue to which half of the stack broke.
    session.spoken.clear()
    session.mcp = None
    asyncio.run(agent._run_llm_turn(session, "what is 12 plus 15?", "c4"))
    texts = spoken_texts(session)
    check("a session with no MCP client still completes the turn",
          len(texts) == 1 and texts[0] == agent.TOOL_FAILURE_TEXT, f"said: {texts}")

    conn = connect_readonly(os.environ["TRACE_DB"])
    dead = [dict(r) for r in conn.execute(
        "SELECT error, json_extract(content, '$.mcp_unavailable') AS mcp_unavailable "
        "FROM events WHERE event_type = 'tool_error' AND json_extract(content, '$.mcp_unavailable') = 1")]
    conn.close()
    check("the missing MCP client is reported as such, not as a NoneType error",
          len(dead) == 1 and "NoneType" not in (dead[0]["error"] or "")
          and "not running" in (dead[0]["error"] or ""),
          str(dead))

    conn = connect_readonly(os.environ["TRACE_DB"])
    # `tool` is a boolean on an llm event and a tool NAME on a tool_error, and
    # timed_out belongs to one event type only, so neither is a promoted column.
    # json_extract reads the blob for them, which is what the content column is
    # for when a question is asked once rather than on every query.
    tool_errors = [dict(r) for r in conn.execute(
        "SELECT json_extract(content, '$.tool') AS tool, "
        "       json_extract(content, '$.timed_out') AS timed_out, error "
        "FROM events WHERE event_type = 'tool_error'")]
    conn.close()
    # Three: a tool that raised, a tool that hung, and a session whose MCP client
    # never started. All three are the same turn-level outcome and all three must
    # be attributable, which is the whole point of recording them separately.
    check("all three tool failures are in the trace", len(tool_errors) == 3, f"{len(tool_errors)}")
    check("a timeout is distinguishable from a failure",
          sorted(e["timed_out"] for e in tool_errors) == [False, False, True],
          f"{[(e['tool'], e['timed_out']) for e in tool_errors]}")
    check("each error says which tool it was",
          all(e["tool"] == "always_fails" for e in tool_errors))

    # --- 5. Piper missing / failing -----------------------------------------
    print("\n5. Piper failing after the LLM already paid for the answer")
    # The real speak() again: section 2 replaced it with a recorder.
    agent.speak = REAL_SPEAK
    session = FakeSession()
    _run_speak_with_piper(session, ["definitely-not-a-real-binary"])
    check("a missing piper does not raise out of the turn", True)
    check("nothing was published", not session.source)

    session = FakeSession()
    _run_speak_with_piper(session, ["/bin/false"])
    check("a non-zero piper exit does not raise out of the turn", True)

    conn = connect_readonly(os.environ["TRACE_DB"])
    tts = [dict(r) for r in conn.execute(
        "SELECT json_extract(content, '$.played') AS played, error "
        "FROM events WHERE event_type = 'tts'")]
    conn.close()
    check("both are recorded as unplayed with a reason",
          len(tts) == 2 and all(not t["played"] and t["error"] for t in tts),
          f"{[t['error'] for t in tts]}")

    # --- 6. whisper.cpp failing ---------------------------------------------
    print("\n6. whisper.cpp crashing mid-call")
    session = FakeSession()
    audio = np.zeros(16000, dtype=np.int16)

    # A binary that exists and fails, which is what a corrupt clip or an
    # unreadable model looks like: diagnostics on stderr, a non-zero exit.
    agent.WHISPER_BIN = "/bin/false"
    asyncio.run(agent.transcribe(session, audio, "c4"))
    check("a non-zero whisper.cpp exit does not raise out of the turn", True)
    check("nothing was spoken, because nothing was transcribed", not session.spoken)

    # A binary that is not there at all.
    session = FakeSession()
    agent.WHISPER_BIN = "/definitely/not/whisper-cli"
    asyncio.run(agent.transcribe(session, audio, "c5"))
    check("a missing whisper binary does not raise out of the turn", True)
    agent.WHISPER_BIN = "/bin/true"

    conn = connect_readonly(os.environ["TRACE_DB"])
    stt = [dict(r) for r in conn.execute(
        "SELECT text, error, json_extract(content, '$.returncode') AS returncode "
        "FROM events WHERE event_type = 'stt'")]
    conn.close()
    check("both STT failures are in the trace with a reason",
          len(stt) == 2 and all(not s["text"] and s["error"] for s in stt),
          f"{[(s['returncode'], s['error']) for s in stt]}")

    # --- 7. the turn-level catch-all ----------------------------------------
    print("\n7. An unforeseen failure inside a turn task")
    session = FakeSession()
    agent.transcribe = _raise_turn
    agent.SESSIONS[session.id] = session
    asyncio.run(agent.handle_turn(session, audio, "c5"))
    agent.SESSIONS.pop(session.id)
    conn = connect_readonly(os.environ["TRACE_DB"])
    failed = [dict(r) for r in conn.execute(
        "SELECT error FROM events WHERE event_type = 'turn_failed'")]
    conn.close()
    check("the turn still left a trace event",
          len(failed) == 1, f"{len(failed)} turn_failed events")
    check("with the traceback in it",
          bool(failed and "RuntimeError" in failed[0]["error"]),
          failed[0]["error"].strip().splitlines()[-1] if failed else "")

    # --- 8. a failing trace write cannot take a turn down -------------------
    print("\n8. Observability failure does not take the pipeline down")
    original = agent.TRACE.log_event

    def exploding_log_event(*args, **kwargs):
        raise RuntimeError("disk full")

    agent.TRACE.log_event = exploding_log_event
    try:
        agent.log_event("s", "c", "stt", {"text": "hi"})
        check("log_event swallows a store failure", True)
    except Exception as e:
        check("log_event swallows a store failure", False, f"raised {type(e).__name__}: {e}")
    finally:
        agent.TRACE.log_event = original

    # --- 9. a model that will not stop talking -------------------------------
    print("\n9. The LLM keeps generating past its answer")
    # Found in a real container run: the tool follow-up came back as
    #   "The sum of 12 plus 15 is 27. **Instruction 2:** <|user|>Given a list ..."
    # so the caller heard their answer and then the model talking to itself. Two
    # things let it through — phi4-mini's chat tokens were not stop sequences,
    # and clean_response's filter anchored on "Instruction" while the model
    # emphasised it as "**Instruction".
    leaks = [
        ("The sum of 12 plus 15 is 27. **Instruction 2:**  \n<|user|>Given a list",
         "The sum of 12 plus 15 is 27."),
        ("It is 48F. <|user|>Now tell me a joke", "It is 48F."),
        ("Sure. **Instruction 1:** answer differently", "Sure."),
        ("Here you go.\nUser: what about tomorrow?", "Here you go."),
        ("Done.\nAgent: I will check that now", "Done."),
        ("Plain and correct.", "Plain and correct."),
        ("Two sentences. They both belong to the answer.", "Two sentences. They both belong to the answer."),
    ]
    for raw, expected in leaks:
        got = agent.clean_response(raw)
        check(f"leak stripped: {raw[:34]!r}...", got == expected, f"got {got!r}")

    check("a leaked instruction cannot survive in the spoken text",
          all("Instruction" not in agent.clean_response(r) for r, _ in leaks))
    check("the chat-template token is a stop sequence for every LLM call",
          "<|user|>" in agent.STOP_SEQUENCES and "<|end|>" in agent.STOP_SEQUENCES,
          str(agent.STOP_SEQUENCES))
    # A two-sentence answer is a real answer, not a leak; truncating it would
    # make the fix cost more than it saves.
    check("a genuine two-sentence answer is not truncated",
          agent.clean_response("It is 48F and raining. Bring an umbrella.") ==
          "It is 48F and raining. Bring an umbrella.",
          agent.clean_response("It is 48F and raining. Bring an umbrella."))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        sys.exit(1)


def _run_speak_with_piper(session, argv):
    """Runs speak() with the piper command replaced by a broken stand-in.

    The command is patched rather than piper's real location, because the point
    is what happens when the binary misbehaves, not that this particular binary
    is missing.
    """
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if cmd and cmd[0] == "piper":
            if argv[0] == "definitely-not-a-real-binary":
                raise FileNotFoundError(2, "No such file or directory", "piper")
            return subprocess.CompletedProcess(cmd, 1, "", "voice file not found")
        return real_run(cmd, **kwargs)

    subprocess.run = fake_run
    try:
        asyncio.run(agent.speak(session, "hello there", "c-tts",
                                {"agent": "primary", "tts_voice": "/bin/true",
                                 "llm_model": "phi4-mini", "tts_voice_overridden": False}))
    finally:
        subprocess.run = real_run


async def _raise_turn(session, audio_data, call_id):
    raise RuntimeError("something nobody predicted")


class _McpStub:
    def __init__(self, mode):
        self.mode = mode

    async def call_tool(self, name, args):
        if self.mode == "hang":
            await asyncio.sleep(30)
        raise RuntimeError("mcp server died")


class _RegistryWithTool:
    """A registry stub whose single tool is always 'always_fails'."""

    tools = {"always_fails": {"description": "Always fails", "args_schema": {}}}
    handoffs = {}

    def build_prompt(self, agent_id, user_text, context_note=""):
        return "You may call always_fails. Answer with JSON."

    def resolve_providers(self, agent_id):
        return {"agent": agent_id, "llm_model": "phi4-mini", "tts_voice": "/bin/true",
                "llm_model_overridden": False, "tts_voice_overridden": False}

    def normalize_args(self, tool_name, args):
        return dict(args or {}), []

    def resolve_handoff(self, tool_name):
        return None

    def get_agent(self, agent_id):
        return {"id": agent_id}


if __name__ == "__main__":
    main()
