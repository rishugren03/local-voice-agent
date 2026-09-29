import asyncio
import os
import wave
import json
import re
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from livekit import rtc, api
import torch
import requests
import time
import uuid
from datetime import datetime

from silero_vad import load_silero_vad
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from livekit.rtc import ParticipantTrackPermission

from agent_platform.config_store import DEFAULT_CONFIG_DB, ConfigStore
from agent_platform.orchestrator import AGENTS_DIR, TOOLS_FILE, AgentRegistry
from agent_platform.trace_store import DEFAULT_DB, TraceStore

load_dotenv()

WHISPER_BIN = os.getenv("WHISPER_BIN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL")
PIPER_MODEL = os.getenv("PIPER_MODEL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4-mini")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
ROOM_NAME = os.getenv("ROOM_NAME", "test-room")

# The LiveKit AudioSource fixes its rate at construction, so this must equal the
# native rate of EVERY voice any agent can publish — validated at startup below.
# Each session gets its own AudioSource, so this one constant covers all of them.
PIPER_SAMPLE_RATE = int(os.getenv("PIPER_SAMPLE_RATE", "22050"))

# whisper.cpp wants 16kHz mono, and the VAD stream is resampled to match.
STT_SAMPLE_RATE = 16000

# Playback is paced in real time (rather than dumped into the audio queue as fast
# as the CPU allows) because barge-in is only observable between frames of a
# real-time loop. This lead keeps a small buffer ahead of the playout clock so
# ordinary scheduling jitter doesn't underrun the stream.
PLAYBACK_LEAD_S = float(os.getenv("PLAYBACK_LEAD_S", "0.12"))

AGENT_RUN_SECONDS = int(os.getenv("AGENT_RUN_SECONDS", "120"))

# Every turn's captured audio is written to chunk_<session>_<call>.wav and deleted
# after whisper reads it. KEEP_CHUNKS=1 keeps them, which is how you find out what
# the VAD actually gated on a turn the trace says should not have happened.
KEEP_CHUNKS = os.getenv("KEEP_CHUNKS")

# Scratch space for those per-turn .wav files. A dedicated directory rather than
# the CWD, because the CWD is not reliably writable: the container runs the agent
# as an unprivileged user against a read-only-ish /app, and a chunk written into
# it raises PermissionError on the first turn — after the caller has already
# spoken, so the turn dies at the one step with no fallback. TMPDIR is honoured so
# a deployment can point it at a volume with the space to spare.
SCRATCH_DIR = os.getenv("AGENT_SCRATCH_DIR") or tempfile.gettempdir()

# Peak |sample| (int16 scale) a turn must reach before it is worth transcribing.
# Measured on real turns: speech peaks at 0.97-1.0 (31000-32767), while the turns
# the VAD gate opens on silence peak at 0. So this sits far from both populations
# and only rejects audio that carries no signal at all.
VAD_PEAK_FLOOR = int(os.getenv("VAD_PEAK_FLOOR", "500"))

# Upstream failure policy. Every stage below (whisper.cpp, Ollama, MCP, Piper) is
# a separate process or service that can be killed mid-conversation, and a voice
# turn has no way to recover: the exception unwinds through the turn task and the
# caller just hears nothing. So each one gets a bounded wait, and each failure
# becomes a spoken fallback plus a trace event naming the cause, rather than a
# turn that silently disappears.
STT_TIMEOUT_S = float(os.getenv("STT_TIMEOUT_S", "30"))
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
LLM_RETRY_BACKOFF_S = float(os.getenv("LLM_RETRY_BACKOFF_S", "1.5"))
MCP_TOOL_TIMEOUT_S = float(os.getenv("MCP_TOOL_TIMEOUT_S", "10"))
PIPER_TIMEOUT_S = float(os.getenv("PIPER_TIMEOUT_S", "30"))

# Canned rather than generated, because the LLM is the thing most likely to be
# down, and a fallback that needs the LLM is no fallback at all. Both are kept to
# one short sentence: these go straight to TTS, so a long one is a long silence.
LLM_FALLBACK_TEXT = "Sorry, I'm having trouble right now. Please try again."
TOOL_FAILURE_TEXT = "Sorry, I couldn't complete that action. Could you rephrase that?"


class LLMUnavailable(RuntimeError):
    """Every attempt to reach the LLM failed.

    Separate from a generic exception because the caller degrades the whole turn
    to LLM_FALLBACK_TEXT on this and nothing else: there is no point in a retry
    loop above this line, and no point asking the model what to say about the
    model being down.
    """


def _discard(path):
    """Removes a per-session temp file if it is there.

    The failure paths below return before the normal cleanup, so without this a
    crashed stage would leave its .wav behind for the whole run. Best-effort by
    design: a file that is already gone is the expected case, not an error.
    """
    try:
        os.remove(path)
    except OSError:
        pass


def _subprocess_error(result, tool):
    """Describe why a subprocess failed, for a log line and the trace.

    A non-zero exit is not always accompanied by a message: whisper.cpp and
    piper can both die on a bad path or a truncated model and exit with nothing
    on stderr. Logging an empty reason makes the failure unsearchable later,
    which is the opposite of what the trace is for, so this falls back to the
    exit code and then to stdout — a failing run's stdout is usually the
    fragment of the real complaint.
    """
    detail = (result.stderr or "").strip() or (result.stdout or "").strip()
    detail = detail[:200]
    if not detail:
        return f"{tool} exited {result.returncode} with no output"
    return f"{tool} exited {result.returncode}: {detail}"


def validate_config():
    required = {
        "WHISPER_BIN": WHISPER_BIN,
        "WHISPER_MODEL": WHISPER_MODEL,
        "PIPER_MODEL": PIPER_MODEL,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise SystemExit(f"Missing required .env values: {', '.join(missing)}")

    for name, path in [("WHISPER_BIN", WHISPER_BIN), ("WHISPER_MODEL", WHISPER_MODEL), ("PIPER_MODEL", PIPER_MODEL)]:
        if not os.path.exists(path):
            raise SystemExit(f"{name} points to a path that doesn't exist: {path}")

    print("[CONFIG] All paths validated.")


validate_config()

DEFAULT_ENTRY_AGENT = "primary"

# Populated in main(); module-level so the helpers below stay plain functions
# rather than methods, matching the single-file layout of this agent process.
REGISTRY = None
ROOM = None
VAD_MODEL = None

# --- Per-session state ---
# SESSIONS is keyed by session_id (one per participant connection, stable across
# every turn that participant takes). This replaces the old single global
# CURRENT_AGENT dict, so multiple simultaneous calls each get their own
# independent active-agent tracking.
SESSIONS = {}

# Indexed by participant identity so a participant that republishes its mic track
# (mute/unmute) rejoins its EXISTING session and keeps its active agent, instead
# of silently starting over on the default one.
SESSION_BY_IDENTITY = {}

# Serializes trace writes; sessions now append from concurrent tasks. The lock
# itself lives in TraceStore, which every other trace reader/writer goes through.
TRACE = TraceStore(os.getenv("TRACE_DB", DEFAULT_DB))

# Same file as the trace database by default, because there is one config and
# one set of call history and splitting them across two files buys nothing here.
# Separate paths stay available through the environment for the case where the
# config wants to be reset or checked out fresh without taking the call history
# with it.
CONFIG_DB = os.getenv("CONFIG_DB", DEFAULT_CONFIG_DB)

# Opened in main() once the paths are settled; None means the worker fell back to
# the JSON files.
CONFIG_STORE = None


@dataclass
class PlaybackState:
    """Barge-in state for ONE session.

    This used to be a single global AGENT_STATE dict shared by every caller, which
    meant one participant talking could set the interrupt flag that another
    participant's playback loop was watching. It is per-session now, so a barge-in
    can only ever truncate the audio of the person who actually interrupted.
    """
    mode: str = "LISTENING"
    interrupt: bool = False
    call_id: str = None


class Session:
    """Everything one participant's call owns: agent state, playback state, its own
    outbound audio track, and its own MCP client. Two concurrent calls touch
    disjoint Session objects and share nothing mutable."""

    def __init__(self, session_id: str, identity: str):
        self.id = session_id
        self.identity = identity
        self.active = DEFAULT_ENTRY_AGENT
        self.context = ""
        self.playback = PlaybackState()
        # Per-session outbound track. A single shared track would deliver every
        # session's answer to every participant in the room, so each session gets
        # its own and routing is enforced by track subscription permissions.
        self.source = rtc.AudioSource(PIPER_SAMPLE_RATE, 1)
        self.track = rtc.LocalAudioTrack.create_audio_track(f"agent-voice-{identity}", self.source)
        self.publication = None
        # Per-session MCP client. Sharing one stdio session across concurrent
        # tool calls means their responses can be interleaved on one pipe.
        # mcp_task owns the client's whole lifetime: the stdio client's anyio
        # cancel scope must be entered and exited by the SAME task, so it is
        # started here and torn down by signalling, never exited from the
        # caller's task.
        self.mcp = None
        self.mcp_task = None
        self.mcp_ready = asyncio.Event()
        self.mcp_stop = asyncio.Event()
        # One turn at a time per session: a barge-in that arrives mid-turn is
        # captured by the VAD loop and processed as the next turn instead of
        # racing the current one.
        self.turn_lock = asyncio.Lock()
        self.vad_task = None
        self.turns_served = 0
        self.interrupts = 0


def get_session(session_id, identity):
    """Fetch or create the session for a participant, keyed by identity so the
    active agent survives a track resubscribe but never crosses participants."""
    if identity in SESSION_BY_IDENTITY:
        return SESSION_BY_IDENTITY[identity]

    session = Session(session_id, identity)
    SESSIONS[session_id] = session
    SESSION_BY_IDENTITY[identity] = session
    TRACE.open_session(session_id, identity)
    print(f"[SESSION] New session {session_id} for '{identity}', starting on '{DEFAULT_ENTRY_AGENT}'")
    return session


def log_event(session_id, call_id, event_type, data):
    """Writes one event to the SQLite trace.

    The call sites are unchanged from the JSONL version — log_event() fills in
    the timestamp and hands the payload to the store, which puts the event, and
    the session/call/agent-transition rows it implies, into the database.

    A failure here is swallowed on purpose. The trace is observability, and an
    observability failure must not take down the turn that is trying to report
    itself; losing an event is strictly better than losing the call, and the
    warning says so loudly enough to notice.
    """
    try:
        TRACE.log_event(session_id, call_id, event_type, data)
    except Exception as e:
        print(f"[ERROR] trace write failed ({event_type}): {type(e).__name__}: {e}")


def call_ollama(prompt, stop=None, max_tokens=150, model=None):
    """One completion from the LLM, with a single retry, over the HTTP API.

    Two attempts and no more: the caller is mid-conversation, and a third attempt
    after two have already failed just extends the silence. The wait is bounded
    by LLM_TIMEOUT_S, and because this runs on a worker thread a hung Ollama
    stalls only this turn — the VAD loop and any other session keep running, so
    barge-in still works while a retry is in flight.

    Everything that can go wrong raises LLMUnavailable: connection refused, read
    timeout, non-2xx, a body with no "response" key, and an empty completion.
    The last two are treated as failures on purpose — an empty turn has no
    utterance to send to TTS, and retrying it is the only thing that can help.
    """
    # model is passed per-call so each agent can use its own LLM; the platform
    # default is only the fallback for callers that don't specify one.
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {
            # Falls back to the platform stop list rather than to nothing, so a
            # call site that forgets to pass one still terminates at the model's
            # turn boundary instead of running to max_tokens and talking to
            # itself.
            "stop": list(stop) if stop else list(STOP_SEQUENCES),
            "num_predict": max_tokens
        }
    }

    last_error = None
    for attempt in (1, 2):
        try:
            response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload,
                                     timeout=LLM_TIMEOUT_S)
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
            if not text:
                raise ValueError("empty completion")
            return text
        except (requests.RequestException, ValueError, KeyError) as e:
            last_error = e
            print(f"[ERROR] LLM attempt {attempt}/2 failed: {type(e).__name__}: {e}")
            if attempt == 1:
                # A short backoff, not an exponential schedule. Worth one: a
                # freshly restarted Ollama answers the first request while it
                # loads the model, and succeeds on the retry.
                time.sleep(LLM_RETRY_BACKOFF_S)

    raise LLMUnavailable(f"{type(last_error).__name__}: {last_error}") from last_error


async def call_ollama_async(prompt, stop=None, max_tokens=150, model=None):
    """call_ollama on a worker thread.

    A bare requests.post blocks the event loop for the whole LLM latency, which
    freezes every session's VAD loop — barge-in stops working and the other
    call's turn latency inflates by this call's LLM time.
    """
    return await asyncio.to_thread(call_ollama, prompt, stop, max_tokens, model)


def clean_response(text):
    """Reduce a model reply to the one or two sentences a caller should hear.

    A 4B local model does not reliably stop where it is asked to. Given an
    instruction it will keep going and start writing the *next* turn: another
    Instruction block, a fake user turn, a fresh "Agent:" line. Left alone, the
    caller hears their answer followed by a machine talking to itself, which is
    worse than no answer because it sounds like the agent is confused.

    Three layers, because each catches what the others miss:
      1. cut at a structural marker (a rule of three, below)
      2. drop sentences that are a new instruction or a fake turn
      3. keep at most the first two sentences

    The markers include phi4-mini's chat-template tokens. It emits
    "<|user|>" to start the next turn, and that is the single most reliable
    boundary it offers — which is why it is also in the stop sequences passed to
    Ollama, so in the common case generation ends there instead of being cleaned
    up afterwards.
    """
    # Structural cut points. Ordered so the earliest real boundary wins.
    text = re.split(
        r'\n---\n|\*\*Note:?\*\*|^Note:|\*\*The following|<\|user\|>|<\|end\|>|'
        r'<\|assistant\|>|\nUser:|\nAgent:',
        text, maxsplit=1)[0].strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept = []
    for sentence in sentences:
        stripped = sentence.strip()
        # Tolerate markdown emphasis around the label. "**Instruction 2:**" is
        # the same leak as "Instruction 2:", and an anchored match on the bare
        # word lets every emphasised version straight through.
        if re.match(r'^\**\s*(Instruction|Task|Prompt|Question|Example)\s*\d*\s*\**\s*:?',
                    stripped, re.IGNORECASE):
            break
        kept.append(stripped)
    return ' '.join(kept[:2]).strip()


# Stop sequences for every LLM call in the pipeline.
#
# One list, used by all three call sites, because the leak this prevents is
# specific to each prompt: the first pass emits a fake "User:" line, the
# clarifying question emits another Instruction block, and the tool follow-up
# emits a whole new task. A stop list maintained per call site is a list where
# one of them eventually forgets the token the model actually uses.
#
# "<|user|>" and "<|end|>" are phi4-mini's chat-template markers. They are the
# reliable ones for this model; the textual ones are kept for a model that was
# fine-tuned on plain "User:" transcripts.
STOP_SEQUENCES = ["\nUser", "User:", "\nAgent", "Agent:", "\n---", "\n###",
                  "<|user|>", "<|end|>", "<|assistant|>", "<|im_end|>"]


async def refresh_route_permissions():
    """Point each session's outbound track at exactly one subscriber.

    LiveKit's TrackPublishOptions has no per-participant destination field in
    this SDK version, so routing is done with track subscription permissions:
    allow nobody by default, then grant each participant its own track SID. The
    server enforces this, so a session's audio is only delivered to the caller it
    belongs to even though every caller shares one room.
    """
    perms = [
        ParticipantTrackPermission(
            participant_identity=s.identity,
            allowed_track_sids=[s.publication.sid],
        )
        for s in SESSIONS.values()
        if s.publication is not None
    ]
    ROOM.local_participant.set_track_subscription_permissions(
        allow_all_participants=False,
        participant_permissions=perms,
    )
    print(f"[ROUTE] {' | '.join(f'{p.participant_identity}<-{list(p.allowed_track_sids)}' for p in perms)}")


async def start_session(participant):
    """Create (or re-attach) a session and publish its dedicated agent track."""
    # Config reload happens here, at the start of a session, and nowhere else.
    # A reload mid-turn would swap the tool list underneath a prompt that was
    # already built from the old one, so a tool call the model just made could
    # come back as unknown — which looks like a model failure and is really a
    # config race. A session is the one moment no prompt is in flight.
    if REGISTRY is not None and REGISTRY.refresh_if_changed():
        print(f"[ROUTE] Picked up a new config for {participant.identity}")

    session = get_session(f"{participant.identity}-{str(uuid.uuid4())[:6]}", participant.identity)

    if session.publication is None:
        session.publication = await ROOM.local_participant.publish_track(session.track)
        print(f"[ROUTE] Published {session.track.name} sid={session.publication.sid} for session {session.id}")
        await refresh_route_permissions()

    if session.mcp_task is None:
        session.mcp_task = asyncio.create_task(mcp_owner(session))
        await session.mcp_ready.wait()

    return session


async def mcp_owner(session):
    """Own one MCP stdio client for a session's whole lifetime.

    Entering and exiting the stdio context from different tasks raises
    'Attempted to exit cancel scope in a different task than it was entered in'
    and leaves the event loop unable to finish, so both happen in here.
    """
    # sys.executable, not "python3".
    #
    # The MCP server is a Python subprocess, and "python3" resolves through PATH,
    # which is not necessarily the interpreter running this file. In a venv that
    # has not been activated — a container CMD of `python transcribe_test.py`,
    # a cron entry, an IDE run configuration — PATH still finds the system
    # python3, which has no `mcp` package installed. The subprocess then dies
    # before it can answer, the client sees the pipe close, and every tool call
    # in the session fails with a transport error that looks like a server
    # problem rather than a PATH problem. sys.executable is the one interpreter
    # guaranteed to have imported the package we are using right now.
    interpreter = os.getenv("MCP_PYTHON") or sys.executable
    script = os.getenv("MCP_SERVER_SCRIPT", "mcp_server.py")
    print(f"[MCP] Starting {script} with {interpreter}")
    ctx = stdio_client(StdioServerParameters(command=interpreter, args=[script]))
    try:
        read, write = await ctx.__aenter__()
        session.mcp = ClientSession(read, write)
        await session.mcp.__aenter__()
        await session.mcp.initialize()
        print(f"[MCP] Session {session.id} has its own MCP client")
        session.mcp_ready.set()
        await session.mcp_stop.wait()
    except Exception as e:
        print(f"[WARN] MCP client error for {session.id}: {e}")
        session.mcp_ready.set()
    finally:
        try:
            if session.mcp is not None:
                await session.mcp.__aexit__(None, None, None)
            await ctx.__aexit__(None, None, None)
        except Exception as e:
            print(f"[WARN] MCP close failed for {session.id}: {e}")
        session.mcp = None


async def teardown_session(identity):
    """Release everything a finished call owns, so a long-lived room does not
    accumulate tracks, MCP subprocesses, and VAD tasks."""
    session = SESSION_BY_IDENTITY.pop(identity, None)
    if session is None:
        return

    if session.vad_task:
        session.vad_task.cancel()

    if session.publication is not None:
        try:
            await ROOM.local_participant.unpublish_track(session.publication.sid)
        except Exception as e:
            print(f"[WARN] unpublish failed for {session.id}: {e}")
        session.publication = None

    session.source.clear_queue()

    if session.mcp_task is not None:
        session.mcp_stop.set()
        try:
            await asyncio.wait_for(session.mcp_task, timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            session.mcp_task.cancel()
        session.mcp_task = None

    SESSIONS.pop(session.id, None)

    log_event(session.id, None, "session_end", {
        "identity": identity,
        "final_agent": session.active,
        "turns_served": session.turns_served,
        "barge_ins": session.interrupts,
    })
    print(f"[SESSION] Ended {session.id} for '{identity}': {session.turns_served} turn(s), "
          f"{session.interrupts} barge-in(s), final agent '{session.active}'")

    await refresh_route_permissions()


async def get_llm_response(session, user_text, call_id):
    """The whole LLM half of a turn, with the LLM-down case handled once.

    The retry lives in call_ollama(); by the time LLMUnavailable reaches here
    both attempts have been spent. There is nothing left to ask the model, so the
    turn degrades to a canned line and a trace event naming the cause. Wrapping
    the whole turn rather than each of its three call sites means a failure in
    the follow-up "state the tool result" call is handled the same way as a
    failure in the first one.
    """
    try:
        await _run_llm_turn(session, user_text, call_id)
    except LLMUnavailable as e:
        print(f"[ERROR] Session {session.id} turn {call_id} degraded to the offline "
              f"fallback: {e}")
        log_event(session.id, call_id, "llm_unavailable", {
            "error": str(e),
            "agent": session.active,
        })
        # providers is left to resolve from the session's active agent, so the
        # fallback is still spoken in the voice that agent is configured with.
        await speak(session, LLM_FALLBACK_TEXT, call_id)


async def _run_llm_turn(session, user_text, call_id):
    t0 = time.monotonic()
    active_agent_id = session.active
    context_note = f"\n(Context from handoff: {session.context})" if session.context else ""

    providers = REGISTRY.resolve_providers(active_agent_id)
    llm_model = providers["llm_model"]
    tts_voice = providers["tts_voice"]
    session.last_llm_model = llm_model
    session.last_tts_voice = tts_voice

    tool_prompt = REGISTRY.build_prompt(active_agent_id, user_text, context_note)

    raw_response = await call_ollama_async(tool_prompt, stop=STOP_SEQUENCES, max_tokens=150, model=llm_model)
    print(f"[TIMING] LLM (first pass) took {time.monotonic() - t0:.2f}s")

    json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    tool_used = False
    tool_name = None
    tool_args = None

    if json_match:
        try:
            call = json.loads(json_match.group())
            if "tool" in call:
                tool_used = True
                tool_name = call["tool"]
                raw_args = call.get("args", {})
                # The tool's args_schema in tools.json is the contract, so the parsed
                # args are cast to their declared types here. A small model sends
                # numbers as strings and occasionally invents keys; either one would
                # otherwise reach the MCP server as-is and fail the call.
                tool_args, arg_notes = REGISTRY.normalize_args(tool_name, raw_args)
                if arg_notes:
                    print(f"[DEBUG] Session {session.id} arg fixes for {tool_name}: {arg_notes}")
                print(f"[DEBUG] Session {session.id} calling tool: {tool_name}({tool_args})")

                handoff_target = REGISTRY.resolve_handoff(tool_name)

                if handoff_target:
                    # Handoff moves THIS session only. session.active is per-session
                    # state, so no other call's active agent can be affected.
                    session.active = handoff_target
                    session.context = tool_args.get("reason", "")
                    target_agent = REGISTRY.get_agent(handoff_target)
                    response = f"Sure, let me connect you with {target_agent['id']}."
                    log_event(session.id, call_id, "handoff", {
                        "from_agent": active_agent_id,
                        "to_agent": handoff_target,
                        "reason": session.context,
                    })
                    print(f"[HANDOFF] Session {session.id}: {active_agent_id} -> {handoff_target}")
                    # Handoff reply uses the NEW target's voice/model on the next turn;
                    # we still log the LLM that produced this handoff utterance.
                else:
                    # Two ways this call must not be sent: the tool is not in the
                    # registry at all (a small model invents tool names), or the
                    # schema says an argument is missing. MCP rejects both, and an
                    # unhandled MCP error here would kill the turn silently, so
                    # either way the turn degrades to a clarifying question.
                    if tool_name not in REGISTRY.tools:
                        reject_notes = [f"'{tool_name}' is not a declared tool"]
                    elif any(n.startswith("missing required") for n in arg_notes):
                        reject_notes = arg_notes
                    else:
                        reject_notes = None

                    if reject_notes:
                        print(f"[WARN] Session {session.id} dropped {tool_name} call: {reject_notes}")
                        log_event(session.id, call_id, "tool_call_rejected", {
                            "tool": tool_name,
                            "raw_args": raw_args,
                            "notes": reject_notes,
                        })
                        response = await call_ollama_async(
                            f"User said: \"{user_text}\". You do not have enough information to use "
                            f"your tools for that. Ask one short clarifying question.\nAgent:",
                            stop=STOP_SEQUENCES, max_tokens=40,
                            model=llm_model
                        )
                    else:
                        # A dead client is checked before the call rather than
                        # caught after it. session.mcp is None when the server
                        # never started, and calling through it raises
                        # "AttributeError: 'NoneType' object has no attribute
                        # 'call_tool'" — which the handler below would catch, but
                        # it names a Python internal instead of the actual fault,
                        # and that is the one thing an operator reading this log
                        # needs to be told.
                        if session.mcp is None:
                            reason = "the MCP tool server is not running for this session"
                            print(f"[ERROR] Session {session.id} cannot run "
                                  f"{tool_name}: {reason}")
                            log_event(session.id, call_id, "tool_error", {
                                "tool": tool_name,
                                "args": tool_args,
                                "error": reason,
                                "timed_out": False,
                                "mcp_unavailable": True,
                            })
                            # Spoken directly rather than through a follow-up LLM
                            # call: the tool is what failed, and the caller
                            # deserves to be told plainly rather than have the
                            # model dress a failure up as an answer.
                            response = TOOL_FAILURE_TEXT
                        else:
                            # The MCP client raises a wide, version-dependent set
                            # of types here (transport closure, protocol error, a
                            # tool that raised, a stdio server that died), so the
                            # handler is deliberately broad: the contract is that
                            # no tool failure takes the turn down, and the trace
                            # event says which kind it was. A timeout is separated
                            # out because a hung tool and a broken one need
                            # different fixes.
                            try:
                                tool_result = await asyncio.wait_for(
                                    session.mcp.call_tool(tool_name, tool_args),
                                    timeout=MCP_TOOL_TIMEOUT_S)
                            except Exception as e:
                                timed_out = isinstance(e, asyncio.TimeoutError)
                                print(f"[ERROR] Session {session.id} tool {tool_name} "
                                      f"{'timed out' if timed_out else 'failed'}: "
                                      f"{type(e).__name__}: {e}")
                                log_event(session.id, call_id, "tool_error", {
                                    "tool": tool_name,
                                    "args": tool_args,
                                    "error": f"{type(e).__name__}: {e}",
                                    "timed_out": timed_out,
                                })
                                # Spoken directly rather than through a follow-up
                                # LLM call: the tool is what failed, and the caller
                                # deserves to be told plainly rather than have the
                                # model dress a failure up as an answer.
                                response = TOOL_FAILURE_TEXT
                            else:
                                result_text = tool_result.content[0].text if tool_result.content else "No result"
                                print(f"[DEBUG] Tool result: {result_text}")

                                t1 = time.monotonic()
                                response = await call_ollama_async(
                                    f"The user asked: \"{user_text}\". "
                                    f"You used the tool {tool_name} with args {tool_args}, and it returned: {result_text}. "
                                    f"Respond to the user naturally with this result, in one short sentence. "
                                    f"Do not invent extra context (like IDs, sessions, etc.) — just state the answer.\nAgent:",
                                    stop=STOP_SEQUENCES, max_tokens=60,
                                    model=llm_model
                                )
                                print(f"[TIMING] LLM (follow-up) took {time.monotonic() - t1:.2f}s")

        except (json.JSONDecodeError, KeyError):
            tool_used = False

    if not tool_used:
        response = raw_response

    response = clean_response(response)
    # `agent` is who SPOKE this turn; on a handoff the session has already moved
    # on, so next_agent records where the conversation went instead.
    print(f"[DEBUG] Session {session.id} speaking as '{active_agent_id}' -> next '{session.active}' (model={llm_model}, voice={os.path.basename(tts_voice)})")
    print(f"Agent[{session.id}]: {response}")
    elapsed = time.monotonic() - t0
    log_event(
        session.id,
        call_id,
        "llm",
        {
            "duration_s": elapsed,
            "response": response,
            "tool": tool_used,
            # Which tool fired and with what args, so a turn can be audited from the
            # trace without re-running the LLM (and so a config-only tool shows up
            # here the same as a hardcoded one).
            "tool_name": tool_name if tool_used else None,
            "tool_args": tool_args if tool_used else None,
            "agent": active_agent_id,
            "next_agent": session.active,
            "llm_model": llm_model,
            "tts_voice": os.path.basename(tts_voice),
            "llm_model_overridden": providers["llm_model_overridden"],
            "tts_voice_overridden": providers["tts_voice_overridden"],
        },
    )
    await speak(session, response, call_id, providers)


async def run_vad(session, track: rtc.Track):
    """Per-session VAD loop.

    This is the session's only always-on task: it must keep consuming audio
    while the turn pipeline runs, otherwise barge-in is impossible. It reads and
    writes ONLY session.playback, so it can never disturb another call.
    """
    stream = rtc.AudioStream(track, sample_rate=STT_SAMPLE_RATE, num_channels=1)

    window_size = 512
    rolling_buffer = np.array([], dtype=np.int16)
    speech_buffer = []
    is_speaking = False
    silence_frames = 0
    SILENCE_THRESHOLD = 20

    async for event in stream:
        frame = event.frame
        # .copy() is load-bearing: frame.data is a memoryview into a buffer the
        # LiveKit SDK reuses and overwrites for every event, so a bare
        # np.frombuffer() is a live view of audio that no longer exists by the
        # time the next event lands.
        data = np.frombuffer(frame.data, dtype=np.int16).copy()
        rolling_buffer = np.concatenate([rolling_buffer, data])

        while len(rolling_buffer) >= window_size:
            chunk = rolling_buffer[:window_size]
            rolling_buffer = rolling_buffer[window_size:]

            float_chunk = chunk.astype(np.float32) / 32768.0
            speech_prob = VAD_MODEL(torch.from_numpy(float_chunk), STT_SAMPLE_RATE).item()

            if speech_prob > 0.5:
                # Read/write only this session's playback state: the interrupt flag
                # is raised here and consumed by this session's publish loop.
                if session.playback.mode == "SPEAKING" and not session.playback.interrupt:
                    session.playback.interrupt = True
                    session.interrupts += 1
                    print(f"[BARGE-IN] Session {session.id} interrupted call {session.playback.call_id}")
                    log_event(session.id, session.playback.call_id, "barge_in", {
                        "speech_prob": round(speech_prob, 3),
                        "active_agent": session.active,
                    })
                is_speaking = True
                silence_frames = 0
                speech_buffer.append(chunk)

            elif is_speaking:
                silence_frames += 1
                speech_buffer.append(chunk)

                if silence_frames >= SILENCE_THRESHOLD:
                    full_audio = np.concatenate(speech_buffer)
                    speech_buffer = []
                    is_speaking = False
                    silence_frames = 0

                    # Silero's gate is a single frame over 0.5, and under load
                    # (two callers sharing one CPU) it occasionally fires on
                    # digital silence. The turn it opens is 0.67s of zeros, and
                    # whisper.cpp answers that with a hallucinated word rather
                    # than an empty string, so the guard below never sees "no
                    # text" — it goes on to burn an LLM call and a TTS render on
                    # nothing. Real speech peaks an order of magnitude higher
                    # than this floor, so refusing the turn costs no real one.
                    peak = int(np.abs(full_audio).max()) if full_audio.size else 0
                    if peak < VAD_PEAK_FLOOR:
                        call_id = str(uuid.uuid4())[:8]
                        print(f"[DEBUG] Session {session.id} dropped a silent "
                              f"{len(full_audio) / STT_SAMPLE_RATE:.2f}s turn (peak={peak})")
                        log_event(session.id, call_id, "vad_rejected", {
                            "reason": "audio below peak floor",
                            "peak": peak,
                            "peak_floor": VAD_PEAK_FLOOR,
                            "audio_s": round(len(full_audio) / STT_SAMPLE_RATE, 3),
                        })
                        continue

                    call_id = str(uuid.uuid4())[:8]
                    print(f"[DEBUG] Session {session.id} end of turn, transcribing (call {call_id})...")
                    # Background task: the VAD loop must stay free to hear a
                    # barge-in while this turn is being transcribed and answered.
                    asyncio.create_task(handle_turn(session, full_audio, call_id))


async def handle_turn(session, audio_data, call_id):
    """Serialize a session's turns. Barge-in speech captured mid-turn is processed
    as the following turn rather than racing the current one."""
    async with session.turn_lock:
        if session.id not in SESSIONS:
            return  # call ended while this turn was queued
        session.turns_served += 1
        try:
            await transcribe(session, audio_data, call_id)
        except Exception:
            # This coroutine runs in a bare create_task, and an exception escaping
            # a task is not reported anywhere until the task object is collected —
            # so a turn that died unexpectedly left no console line and no trace
            # event, which is the "turn silently disappeared" failure. The known
            # upstream failures degrade to a spoken fallback on their own; this is
            # the last resort for what is genuinely unforeseen, and its job is to
            # leave a record with enough of the traceback to act on.
            error = traceback.format_exc(limit=8)
            print(f"[ERROR] Session {session.id} turn {call_id} failed:\n{error}")
            log_event(session.id, call_id, "turn_failed", {"error": error})


async def transcribe(session, audio_data, call_id):
    t0 = time.monotonic()
    wav_path = os.path.join(SCRATCH_DIR, f"chunk_{session.id}_{call_id}.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setparams((1, 2, STT_SAMPLE_RATE, 0, "NONE", "NONE"))
        wf.writeframes(audio_data.tobytes())

    # whisper.cpp is a blocking subprocess; on a thread so the other sessions'
    # VAD loops keep running while this one transcribes.
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav_path, "-nt"],
            capture_output=True, text=True, timeout=STT_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError) as e:
        # A missing or non-executable binary, or a model load that never
        # finishes. Without this the turn died here, before any speech was
        # attempted, and the caller heard nothing at all.
        print(f"[ERROR] Session {session.id} whisper.cpp failed: {type(e).__name__}: {e}")
        log_event(session.id, call_id, "stt", {
            "duration_s": round(time.monotonic() - t0, 3),
            "text": "", "error": f"{type(e).__name__}: {e}",
        })
        _discard(wav_path)
        return

    text = result.stdout.strip()
    elapsed = time.monotonic() - t0
    print(f"[TIMING] STT took {elapsed:.2f}s (session={session.id})")

    if result.returncode != 0:
        # whisper.cpp writes diagnostics to stderr and can exit non-zero on a
        # corrupt clip or an unreadable model. Its stdout is then either empty or
        # a fragment, and transcribing that fragment would answer the user with
        # half a word.
        reason = _subprocess_error(result, "whisper.cpp")
        print(f"[ERROR] Session {session.id} {reason}")
        log_event(session.id, call_id, "stt", {
            "duration_s": elapsed, "text": "", "returncode": result.returncode,
            "error": reason,
        })
        _discard(wav_path)
        return

    log_event(session.id, call_id, "stt", {"duration_s": elapsed, "text": text})

    if not KEEP_CHUNKS:
        _discard(wav_path)  # per-session temp file, clean up after use

    if text:
        print(f"You said[{session.id}]: {text}")
        await get_llm_response(session, text, call_id)


async def speak(session, text, call_id, providers=None):
    # providers is passed in from the turn that generated the text so the voice
    # matches the agent that actually spoke. On a handoff turn the outgoing
    # agent says the connecting line, and the new voice takes over next turn.
    t0 = time.monotonic()
    if providers is None:
        providers = REGISTRY.resolve_providers(session.active)
    tts_voice = providers["tts_voice"]
    llm_model = providers["llm_model"]

    output_wav = os.path.join(SCRATCH_DIR, f"response_{session.id}_{call_id}.wav")
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["piper", "--model", tts_voice, "--output_file", output_wav],
            input=text, text=True, capture_output=True, timeout=PIPER_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError) as e:
        # FileNotFoundError is piper not on PATH; TimeoutExpired is a render that
        # never finished. Both used to unwind out of the turn, discarding an
        # answer the LLM had already paid for. The caller hears nothing either
        # way, but the trace now says which of the two happened.
        error = f"{type(e).__name__}: {e}"
        print(f"[ERROR] Session {session.id} piper failed: {error}")
        log_event(session.id, call_id, "tts", {
            "duration_s": round(time.monotonic() - t0, 3), "returncode": None,
            "agent": providers["agent"], "next_agent": session.active,
            "llm_model": llm_model, "tts_voice": os.path.basename(tts_voice),
            "tts_voice_overridden": providers["tts_voice_overridden"],
            "played": False, "error": error,
        })
        _discard(output_wav)
        return

    elapsed = time.monotonic() - t0
    print(f"[TIMING] TTS synthesis took {elapsed:.2f}s (session={session.id}, voice={os.path.basename(tts_voice)})")
    print(f"[DEBUG] piper returncode: {result.returncode}")

    publish = None
    if result.returncode == 0 and os.path.exists(output_wav):
        publish = await publish_audio(session, call_id, output_wav)
    else:
        # Piper reports why on stderr, and "failed to produce output" on its own
        # does not distinguish a missing voice file from a bad model path — the
        # two need different fixes, so the reason is carried into the trace.
        # A zero exit with no file is its own case: piper ran fine and still
        # wrote nothing, which points at the voice rather than at piper, so it
        # keeps its own wording instead of borrowing the crash message.
        if result.returncode != 0:
            error = _subprocess_error(result, "piper")
        else:
            error = "piper exited 0 but produced no output file"
        print(f"[ERROR] Session {session.id} {error}")
        log_event(session.id, call_id, "tts", {
            "duration_s": elapsed, "returncode": result.returncode,
            "agent": providers["agent"], "next_agent": session.active,
            "llm_model": llm_model, "tts_voice": os.path.basename(tts_voice),
            "tts_voice_overridden": providers["tts_voice_overridden"],
            "played": False, "error": error,
        })
        _discard(output_wav)
        return

    _discard(output_wav)

    log_event(
        session.id,
        call_id,
        "tts",
        {
            "duration_s": elapsed,
            "returncode": result.returncode,
            "agent": providers["agent"],
            "next_agent": session.active,
            "llm_model": llm_model,
            "tts_voice": os.path.basename(tts_voice),
            "tts_voice_overridden": providers["tts_voice_overridden"],
            "played": True,
            "playback_s": publish["playback_s"],
            "playback_end_reason": publish["end_reason"],
            "publish_start": publish["start"],
            "publish_end": publish["end"],
        },
    )


async def publish_audio(session, call_id, wav_path):
    """Stream one turn's audio into THIS session's own track, paced in real time
    so a barge-in can land mid-sentence.

    Returns the publish window and how playback ended, so the trace records which
    turn was audible when — that is what makes cross-session leakage checkable
    from the trace database after the fact.
    """
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    samples = np.frombuffer(audio_data, dtype=np.int16)
    frame_size = 480
    frame_duration = frame_size / sr

    session.playback.mode = "SPEAKING"
    session.playback.interrupt = False
    session.playback.call_id = call_id

    start_iso = datetime.now().isoformat()
    t0 = time.monotonic()
    end_reason = "completed"
    next_frame_at = t0

    for i in range(0, len(samples), frame_size):
        if session.playback.interrupt:
            end_reason = "barge_in"
            break
        chunk = samples[i:i + frame_size]
        frame = rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=sr,
            num_channels=1,
            samples_per_channel=len(chunk)
        )
        await session.source.capture_frame(frame)

        next_frame_at += frame_duration
        delay = next_frame_at - time.monotonic() - PLAYBACK_LEAD_S
        if delay > 0:
            await asyncio.sleep(delay)

    if end_reason == "barge_in":
        # Frames already queued would keep playing after the interrupt, so drop
        # them. Scoped to this session's own source.
        session.source.clear_queue()
        print(f"[DEBUG] Session {session.id} playback interrupted by barge-in")

    await session.source.wait_for_playout()
    session.playback.mode = "LISTENING"
    end_iso = datetime.now().isoformat()

    return {
        "start": start_iso,
        "end": end_iso,
        "playback_s": round(time.monotonic() - t0, 3),
        "end_reason": end_reason,
    }


async def main():
    global REGISTRY, ROOM, VAD_MODEL, CONFIG_STORE

    # SQLite when it is available, JSON files otherwise.
    #
    # The database is what the control plane writes to, so it is what makes
    # "edit the agent in the browser, hear it on the next call" true. The file
    # fallback is not a compatibility shim: it means the worker still starts and
    # answers calls when the database has not been created yet, which is the
    # state of a fresh checkout, and a control plane that will not boot because
    # nobody ran an import first is a control plane nobody runs.
    try:
        CONFIG_STORE = ConfigStore(CONFIG_DB)
        # Seed once per database, decided by a marker row rather than by
        # emptiness — same helper the control plane calls, so the worker and the
        # API cannot disagree about when a database "has no config". A user who
        # deletes the last agent in the UI means it, and this must not undo that
        # on the next start.
        if CONFIG_STORE.seed_from_json_if_empty(AGENTS_DIR, TOOLS_FILE):
            print("[CONFIG] Database has no agents yet; imported the JSON config.")
    except Exception as e:
        print(f"[WARN] Config store unavailable ({e}); falling back to the JSON files.")
        CONFIG_STORE = None

    REGISTRY = AgentRegistry(
        default_llm_model=OLLAMA_MODEL,
        default_tts_voice=PIPER_MODEL,
        config_store=CONFIG_STORE,
    )
    source = "sqlite" if CONFIG_STORE is not None else "json files"
    print(f"[CONFIG] Loaded {len(REGISTRY.agents)} agents and "
          f"{len(REGISTRY.tools)} tools from {source}.")

    print("[DEBUG] Warming up LLM...")
    # The warmup only exists to move the model's first-load cost off the first
    # real turn. Failing it must not stop the agent from joining the room: a
    # caller should still get a connected agent that says its fallback line while
    # Ollama is down, which is the entire point of the degradation path.
    try:
        call_ollama("Say OK.", max_tokens=5)
    except LLMUnavailable as e:
        print(f"[WARN] LLM warmup failed ({e}). Turns will fall back until it recovers.")
    print("[DEBUG] LLM warm.")

    print("[DEBUG] Loading VAD...")
    VAD_MODEL = load_silero_vad()
    print("[DEBUG] VAD ready.")

    url = os.getenv('LIVEKIT_URL')
    token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
        .with_identity("voice-agent") \
        .with_name("Voice Agent") \
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME)) \
        .to_jwt()

    ROOM = rtc.Room()

    @ROOM.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # session_id is stable per participant, generated once — this is what
            # makes agent state (which persona is active) persist across that
            # participant's turns without leaking into other participants' calls.
            asyncio.create_task(attach_track(track, participant))

    @ROOM.on("participant_disconnected")
    def on_participant_disconnected(participant):
        print(f"[SESSION] {participant.identity} disconnected")
        asyncio.create_task(teardown_session(participant.identity))

    await ROOM.connect(url, token)
    print(f"Agent joined room: {ROOM.name}")

    try:
        voice_report = REGISTRY.validate_voice_sample_rates(PIPER_SAMPLE_RATE)
        print("[CONFIG] Voice sample-rate validation passed:")
        for row in voice_report:
            print(
                f"  [{row['source']}] {row['voice']} @ {row['sample_rate']} Hz (matches_track={row['matches_track']})"
            )
    except ValueError as e:
        print("[ERROR] Voice sample-rate validation failed:", e)
        await ROOM.disconnect()
        raise

    print("Join the room from any number of tabs — each participant gets its own "
          "agent session, its own active agent, and audio only they can hear.")

    try:
        await asyncio.sleep(AGENT_RUN_SECONDS)
    finally:
        for identity in list(SESSION_BY_IDENTITY.keys()):
            await teardown_session(identity)
        await ROOM.disconnect()
        # Committed and closed explicitly, so the last session_end is on disk
        # before the process exits rather than waiting on interpreter teardown.
        TRACE.close()
        print(f"[TRACE] Wrote trace to {TRACE.path}")


async def attach_track(track, participant):
    session = await start_session(participant)
    if session.vad_task is None or session.vad_task.done():
        session.vad_task = asyncio.create_task(run_vad(session, track))
    print(f"[SESSION] {participant.identity} attached to session {session.id} (agent='{session.active}')")


if __name__ == "__main__":
    asyncio.run(main())
