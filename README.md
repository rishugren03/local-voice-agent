<p align="center">
  <img src="assets/logo.svg" alt="" width="88" height="88" />
</p>

# Local Voice Agent — Fully Offline Voice AI Orchestration

A self-hosted voice agent pipeline — the same architectural pattern as Vapi/LiveKit Agents — but with **zero cloud dependency in the hot path**. Every stage (transport, STT, LLM reasoning, tool calling, TTS) runs entirely on-device.

## Why local-first

Cloud voice AI platforms (Vapi, Bland, Retell) are built on hosted STT/LLM/TTS APIs. That's the right call for most products, but it means every conversation leaves the building and costs money per minute. This project explores the opposite end of the tradeoff space: **can a voice agent run entirely on local hardware, with acceptable latency, while still supporting the features production voice agents need** — interruption handling, tool use, multi-agent handoff?

This isn't a replacement for Vapi. It's a demonstration that the orchestration patterns those platforms use aren't tied to the cloud — they can run on a laptop, with real privacy and cost benefits for compliance-sensitive use cases (healthcare, legal, defense).

## Architecture

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────┐     ┌───────┐
│   LiveKit   │────▶│  Silero  │────▶│ whisper.cpp │────▶│ Phi-4-   │────▶│ Piper │
│ (transport) │     │   VAD    │     │    (STT)    │     │ mini     │     │ (TTS) │
└─────────────┘     └──────────┘     └─────────────┘     │ (LLM)    │     └───────┘
       ▲                  │                                └────┬────┘         │
       │                  │ turn-taking +                       │              │
       │                  │ barge-in signal                     ▼              │
       │                                                  ┌──────────┐         │
       │                                                  │   MCP    │         │
       │                                                  │  tools   │         │
       │                                                  └──────────┘         │
       └─────────────────────────────────────────────────────────────────────┘
                              audio published back into room
```

**Stack:**
| Component | Choice | Why |
|---|---|---|
| Transport | LiveKit (self-hosted, dev mode) | Industry-standard WebRTC infra; don't reinvent media transport |
| VAD | Silero VAD | Lightweight, accurate speech/silence detection, runs per-frame |
| STT | whisper.cpp (`base.en`) | Fully local, CPU-friendly, no API cost |
| LLM | Phi-4-mini via Ollama | Small enough for real-time local inference, capable enough for tool-use reasoning |
| TTS | Piper (`en_US-lessac-medium`) | Fast local synthesis, no GPU required |
| Tool calling | MCP (Model Context Protocol) | Standard protocol for agent-tool integration, not custom glue |
| Multi-agent | Prompt-based persona switching | Simple, effective handoff mechanism for a small local model |

## Key design decisions

**Turn-taking is VAD-driven, not timer-based.** Early versions used a fixed 4-second buffer-then-transcribe loop. This was replaced with Silero VAD running on a continuous rolling window, triggering transcription only after ~640ms of silence following detected speech. This is what makes the agent feel responsive rather than sluggish.

**Barge-in requires true concurrency.** The hardest bug in this project: naive sequential code (`await transcribe() → await respond() → await speak()`) blocks the VAD loop while the agent is talking, so it can never detect an interruption. Fixed by making the transcribe→respond→speak chain a background `asyncio.create_task`, freeing the VAD loop to keep listening continuously — including while the agent is speaking. This is the actual mechanism, not a hack: the agent state machine (`LISTENING` / `SPEAKING`) and an interrupt flag checked between every published audio frame is what makes clean mid-sentence cutoff possible.

**One room, many isolated calls.** Multiple participants can join the same room at once, each with an independent agent session. Everything mutable is owned by a `Session` object keyed by participant identity: the active agent, the handoff context, the barge-in state, the outbound audio track, the MCP client, and a turn lock. A handoff in one call cannot move another call's agent, and one caller talking cannot interrupt another caller's audio. Routing each session's answer to only the caller who asked is enforced with LiveKit track subscription permissions — one dedicated audio track per session, with the server granting each participant access to exactly one of them.

Two things about concurrency are easy to get wrong and are worth stating explicitly, because both were real bugs here:

- **Audio has to be published in real time.** Dumping a synthesized response into the audio queue as fast as the CPU allows finishes in ~0.1s and overruns the source's buffer, so the caller hears a fraction of the answer and no barge-in can be observed — the interrupt flag is checked between frames of a loop that already ended. Playback is paced against the playout clock and the queued audio is dropped on interrupt.
- **The event loop must never block on inference.** whisper.cpp, Piper, and the Ollama HTTP call are all blocking. Called directly they freeze every session's VAD loop for the duration, so barge-in dies and the other call's turn latency inflates by this call's LLM time. They run on worker threads.

**Tool calls go through MCP, not custom function-calling glue.** Since Phi-4-mini has no native function-calling support (unlike GPT-4/Claude), tool invocation is done via a structured JSON-in-prompt convention, parsed and dispatched to a real MCP server. This keeps the tool layer swappable and standards-based rather than model-specific.

**What a tool takes is declared in `tools.json`, not inferred in Python.** Because the tool convention is prompt-based, the prompt has to state each tool's signature — and the first version hardcoded that in the orchestrator, one `if` per tool, so a new tool meant editing the Python. Each tool now carries an `args_schema` in `tools.json` and the prompt (both the tool listing and the JSON example) is generated from it, so the registry contains no tool names at all. The same schema is then used on the way back in: a small model asked for a number usually sends `"3"` as a string and sometimes invents extra keys, so the parsed args are cast to their declared types before the call reaches MCP, and a call missing a required argument is answered with a clarifying question instead of a failed MCP call. Misconfiguration is caught at load — a typo in an agent's `tools` list used to be skipped silently, quietly leaving that agent with fewer tools than its config claimed.

**Small local models need explicit, capitalized rules, not soft suggestions.** During eval testing, Phi-4-mini inconsistently used the `calculate` tool for simple arithmetic — sometimes computing (and getting wrong) answers itself. Softly worded prompts ("if the request needs a tool...") weren't reliable. An explicit rule ("RULE: For ANY math question... you MUST use the calculate tool") fixed this, at the cost of a transient over-triggering regression on unrelated queries during tuning — a real small-model prompt-engineering tradeoff, not a clean fix.

## Adding an agent or a tool

Agents are JSON files in `agent_platform/agents/`; the tool registry is `agent_platform/tools.json`. Neither requires touching Python.

```jsonc
// agent_platform/tools.json — one entry per tool
"get_weather": {
  "description": "Gets the current weather for a city or place",
  "args_schema": {
    "location": {
      "type": "string",                        // string|number|integer|boolean|array|object
      "description": "A city or place, like 'Seattle, WA'",
      "example": "Seattle, WA",                // optional; shown to the LLM
      "required": true                         // optional, defaults to true
    }
  }
}
```

```jsonc
// agent_platform/agents/primary.json — which tools this agent may use
"tools": ["calculate", "check_calendar", "get_weather"]
```

Three steps, in this order:

1. **Implement the tool in `mcp_server.py`** — the actual work, and the one part that is code. `@mcp.tool()` with a typed signature.
2. **Declare it in `tools.json`** with an `args_schema`. The description is what the LLM reads to decide, so it says when to use the tool; each argument needs a `type` and a `description` describing the expected value.
3. **Attach it to an agent's `tools` list.** Handoffs are already declared: an agent's `handoffs` list plus a matching `handoff_to_<id>` entry in `tools.json` needs no per-handoff code.

`example` is optional — an argument with only `type` and `description` falls back to using the description as the placeholder value, which works but gives the model a weaker hint. A standard JSON Schema object (`{"type": "object", "properties": {...}, "required": [...]}`) is accepted in place of the flat form.

For **agents** there is a CLI, so the JSON never has to be written by hand:

```bash
python3 config_cli.py create-agent     # prompts for id, prompt, tools, handoffs, rules, model/voice
python3 config_cli.py edit-agent trivia
python3 config_cli.py list-agents      # table of agents, their tools and handoff targets
python3 config_cli.py validate         # same checks as startup, without starting the pipeline
```

It runs the same `AgentRegistry` the pipeline loads, so a config that passes `validate` is one `transcribe_test.py` can start with. Selecting a handoff whose `handoff_to_<id>` entry is missing from `tools.json` offers to add that entry, because the alternative is a config the registry refuses to load. Tools still have to be declared in `tools.json` by hand — that is step 2 above, and only step 1 is code.

Verify a config change without a live call:

```bash
python3 config_cli.py validate         # config + tool references + voice sample rates
python3 -m agent_platform.orchestrator   # prints every agent's prompt + the rendered tool examples
python3 test_session_isolation.py         # per-session invariants
```

A malformed schema, an unknown tool name, or a handoff with no matching tool fails at startup with the offending entry named, rather than turning into a wrong prompt mid-call.


## Reliability data (eval harness)

A scripted 7-case test suite (greeting, simple math, complex math, calendar tool lookup, weather lookup, multi-agent handoff trigger, off-topic robustness) run against a simulated user (pre-synthesized audio injected into the LiveKit room), scored automatically against expected tool usage and response content. `weather_check` is the regression case for the schema-driven tool registry: that tool exists only in `tools.json` and one agent's `tools` list, so the case passing end to end is the evidence that a config-only tool is offered, parsed, and dispatched.

| Metric | Before prompt fix | After prompt fix |
|---|---|---|
| Task pass rate (6-case suite) | 4/6 | **6/6** |
| Known failure modes | Unreliable math tool-triggering, handoff not firing | None observed in this run |

Latest run, after adding the config-only `weather_check` case (7 cases, default config): **6/7**, with `weather_check` passing and `handoff_trigger` failing. That case is the flaky one — across repeated runs the model either hands off or refuses conversationally, which is why `SCOPE_RULE=1` exists as a knob to push it. An A/B over 10 samples per config put the hand-off rate at 3/10 *without* the weather tool configured and 5/10 *with* it, so the flakiness is the 3B model, not the tool count.

`score_eval.py` now checks the *specific* tool named in the trace rather than "some tool fired". That is what makes a config-only tool verifiable — and it also fails turns that answer a scheduling request with `check_calendar` and a hallucinated date, which the old check scored as a pass.

With two simulated callers in the room at once (`run_eval.py --users 2`), the same suite scored **13/14** — 6/7 and 7/7 for the two callers, interleaved on one CPU. The suite assumes one linear conversation on the default agent, so `--rotate` (which deliberately reorders the cases and can put a handoff in front of later cases) is for stress-testing attribution, not for reading a pass rate off: a caller that has legitimately been handed off to another agent will be answered by that agent.

### Two callers surface two bugs one caller never does

Running two callers at once is what made both of these visible; neither reproduced with `--users 1`.

**A VAD gate that fires on silence.** Under two-caller load the Silero gate (`speech_prob > 0.5` on a single 512-sample frame) occasionally opened a turn on digital silence. Keeping the captured audio (`KEEP_CHUNKS=1`) showed those turns were 0.67s of exact zeros — and `whisper.cpp` answers pure silence with a hallucinated word rather than an empty string, so the existing `if text:` guard never caught them. Each one cost a full LLM call and TTS render on nothing, and appeared in the score as a call that heard `"you"`. The fix is a peak-amplitude floor (`VAD_PEAK_FLOOR`, default 500 on the int16 scale) checked when the turn closes: real speech peaks at 31000-32767, the phantom turns peaked at 0-3, so the floor rejects only audio carrying no signal. Dropped turns are logged as `vad_rejected` and reported per session by the scorer rather than silently discarded.

**A reused audio buffer.** `np.frombuffer(frame.data, ...)` was returning a live view into a buffer the LiveKit SDK overwrites on every frame — verified directly, not inferred: a view of one event's buffer had different contents by the next event. The `.copy()` is load-bearing. It was masked until now because `np.concatenate` happens to copy the rolling buffer on every event, so the corrupt path was narrow; the peak floor above is what made the phantom turns visible enough to go looking.

**Latency (fully local, single consumer machine, no GPU-specific optimization):**
| Stage | p50 | p90 |
|---|---|---|
| STT (whisper.cpp) | 0.87s | 0.89s |
| LLM (Phi-4-mini) | 2.3s | 2.84s |
| TTS (Piper) | 1.2s | 1.37s |
| **Total time-to-response** | **4.53s** | **4.85s** |

The 7-case run above measured STT 1.10s, LLM 2.60s, TTS 1.65s, total 5.51s (p50) / 6.59s (p90) on the same audio-injection harness — the same ballpark, with the gap most likely run-to-run variance on a shared machine. These numbers are from a small eval batch on unoptimized consumer hardware — they represent a baseline, not a ceiling. The LLM stage is the dominant cost; further optimization (smaller quantization, speculative decoding, or streaming token-by-token into TTS) is the clearest path to reducing total latency.

## What's not solved yet

- Latency (~4.5s p50) is well above production voice AI targets (sub-1s is standard for commercial platforms) — this is a known tradeoff of full local inference on consumer hardware, not yet optimized
- One agent process serves one room, so every session shares a CPU with the others; concurrent calls contend for the same local LLM and can each other's latency
- Per-session MCP clients mean one `mcp_server.py` subprocess per connected participant
- Minimal error handling for upstream failures (Ollama crash, Piper failure mid-call)
- STT accuracy on short/noisy utterances is limited by `whisper.cpp base.en` — a larger model would improve this at a latency cost
- A turn the caller made while the agent was still speaking can be captured, queued behind the in-flight turn, and then dropped when the caller disconnects mid-answer: `handle_turn` bails on `session.id not in SESSIONS` (transcribe_test.py), so the last utterance of a call that ends on a slow TTS never gets answered. The eval suite sees it as `MISSING`, which is how it was found
- The probe/analyzer below are a test harness, not a load test: concurrency is verified with two synthetic participants, not at production call volumes

## Running it

```bash
git clone https://github.com/rishugren03/local-voice-agent.git
cd local-voice-agent
cp .env.example .env   # fill in your local paths 
pip install -r requirements.txt

# In separate terminals:
livekit-server --dev
ollama pull phi4-mini
python3 transcribe_test.py
```

Join the room via [meet.livekit.io](https://meet.livekit.io) using `ws://localhost:7880` and a token generated via the LiveKit CLI (see `lk token create` in setup notes).

## Running with Docker

The image builds the whole pipeline itself: whisper.cpp compiled for CPU, CPU-only
torch, Piper, and the agent. Nothing is downloaded at run time except the model
weights, and the `models` service fetches those before the agent starts.

```bash
docker-compose up -d
docker-compose logs -f agent
```

If your Docker has the Compose plugin, `docker compose` works identically; the
examples below use the standalone `docker-compose` binary.

The first build is slow (torch CPU wheels, whisper.cpp); the layers are cached
after that. On a clean machine it pulls roughly 3 GB.

Two things the compose file does that a manual setup usually gets wrong:

- **The whisper binary needs its shared libraries.** `whisper-cli` links against
  `libwhisper` and the ggml backends, which are not in the base image. The
  Dockerfile copies them to `/opt/whisper/lib` and sets `LD_LIBRARY_PATH`; a
  container with only the binary on `PATH` fails at exec with a loader error that
  looks like a corrupt file.
- **Piper voices need their `.onnx.json` sidecar**, which is where the model's
  sample rate lives. Downloading only the `.onnx` produces a voice that renders
  audio at the wrong rate, and playback then fails mid-call with
  `InvalidState: sample_rate and num_channels don't match`. The model service
  fetches both.

Every configured voice is checked against the room's track rate at startup. A
voice at the wrong rate is a startup warning, not a call-time surprise.

If ports 7880 or 11434 are already in use — a `livekit-server --dev` you left
running, or Ollama installed locally — compose will not start, because it binds
them on the host. Either stop the local service, or move the published ports:

```bash
# Only the host side moves; the agent still reaches everything over the compose
# network on the original container ports, so nothing inside needs changing.
LIVEKIT_HTTP_PORT=17880 LIVEKIT_RTC_PORT=17881 OLLAMA_PORT=21434 \
  docker-compose up -d
```

Point anything running outside compose at the shifted ports
(`LIVEKIT_URL=ws://localhost:17880`, `OLLAMA_HOST=http://localhost:21434`).

An override file would not work here, which is why this is an environment
variable: compose *merges* a `ports:` list instead of replacing it, so
`docker-compose.override.yml` can add a port but never remove the one that is
in the way.

## Control plane and web UI

The agent's configuration lives in SQLite (`call_trace.db` by default, same file as
the call history). The control plane is the only thing that writes it; the worker
reads it and reloads when it changes.

```bash
python3 -m control_plane.app            # http://127.0.0.1:8080, docs at /docs
cd ui && npm install && npm run dev     # http://127.0.0.1:5173
```

It has no auth and binds localhost, so anything that can reach the port can mint
a LiveKit token and rewrite any agent prompt. That is a deliberate v1 scope, not
an oversight — do not put it on a shared interface.

| What | Where |
| --- | --- |
| Config CRUD | `GET/PUT/PATCH/DELETE /api/agents`, `/api/tools`, `/api/squads` |
| What the model will see | `GET /api/agents/{id}/prompt-preview?user_text=...` |
| Call history | `GET /api/sessions`, `/api/sessions/{id}`, `/api/calls/{id}`, `/api/timeline` |
| Service checks | `GET /health` — every check that is down carries the command that fixes it |
| Browser join token | `POST /api/calls/token` — the LiveKit secret never leaves the server |
| Eval runs | `GET/POST /api/evals`, `GET /api/evals/status` |
| Import / export | `POST /api/config/import`, `POST /api/config/export` |

Two details worth knowing:

**Validation is shared with the worker.** `agent_platform/validation.py` is the one
implementation of "is this config usable", called both by the API on write and by
the worker on load. Two copies of these rules is how a config gets accepted by the
editor and then rejected by the worker, which surfaces as "the UI said it saved but
the call broke".

**The JSON files are still the durable copy.** The database is the source of truth
during a run; `POST /api/config/export` writes it back to `agent_platform/agents/`
and `tools.json`, which is what `config_cli.py`, the worker on a fresh start, and
git all use. Export before you stop the process, or the next `config_cli.py` run
sees the files as authoritative and your UI edits are gone.

### The web UI

`ui/` is a React + TypeScript SPA built with Vite. Seven screens:

| Screen | What it is for |
| --- | --- |
| **Assistants** | Edit a prompt, its rules, tools and handoffs. Live prompt preview beside the editor. |
| **Talk** | Open a call from the browser, watch the input level and the turn-by-turn transcript with per-stage latency. |
| **Calls** | Every session and turn, with latency bars, handoffs, tool calls, failures, and the raw event stream. |
| **Tools** | Tool descriptions and JSON Schemas, validated as you type. |
| **Squads** | Routing graph with a draggable node layout; missing agents are drawn in red rather than hidden. |
| **Evals** | Start the scripted suite, watch it run, read the score and the runner's own log. |
| **System** | Service status, config validity, and JSON export/import. |

Two decisions shape the whole thing:

**Errors are the backend's words.** Every rejection comes back as a `problems` list
written by `agent_platform/validation.py` or the health checks, and the UI renders
it verbatim. Rewriting those messages client-side would mean two sets of wording to
keep in sync, and the one that drifts is always the copy that has been through a
translation.

**The service banner never goes away.** A down service is the usual reason an agent
sounds worse than its prompt suggests, so it is permanent and it carries the fix,
not just the status.

Build for production with `npm run build` (output in `ui/dist/`). By default the
bundle calls the API on its own origin and the Vite dev server proxies `/api` to
`127.0.0.1:8080`; set `VITE_API_BASE` to point a built bundle at a control plane
somewhere else.

## Concurrency

Every participant in the room gets an independent agent session. To check that by hand, open [meet.livekit.io](https://meet.livekit.io) in **two tabs** with different identities, join the same room, and have each tab talk while the other is mid-answer. Trigger a handoff in one tab (`"can you help me find a free slot next week?"`) while the other keeps asking the default agent math questions (`"what is 12 plus 15?"`). The handoff tab switches to the new agent and its voice; the other tab stays on the default agent, and neither tab hears the other's reply. Interrupting one tab mid-sentence must not cut the other tab off.

`call_trace.db` records the evidence — one query per question, no parsing:

```sql
-- which agent answered, in which voice, for which caller
SELECT session_id, agent, tts_voice, COUNT(*) FROM events WHERE event_type = 'llm' GROUP BY 1, 2, 3;
-- every agent transition in one session
SELECT from_agent, to_agent, reason FROM agent_transitions WHERE session_id = ? ORDER BY ts;
```

### Automated check

```bash
# fast, no services beyond a valid .env — state-machine level invariants
python3 test_session_isolation.py
python3 test_scoring.py           # scorer attribution, on a synthetic interleaved trace
python3 test_degradation.py       # one dependency failing at a time
python3 test_control_plane.py     # API, validation, import/export round trip

# end to end: two synthetic participants join the room, speak overlapping
# scripts, and every participant meters the audio it actually receives
python3 transcribe_test.py          # terminal 1
python3 concurrency_probe.py --out after.json   # terminal 2
python3 analyze_concurrency.py after.json
```

The probe subscribes to every agent track and records per-250ms RMS energy, so "did session A's answer leak into session B's ears" is a measurement rather than an impression. `analyze_concurrency.py` reports audio that appeared on a track outside that listener's own turns (a leak) alongside turns the listener never heard (a regression in the other direction — without that control, muting every caller would also score zero leaks).

## Eval harness

```bash
python3 generate_test_audio.py   # creates synthetic test utterances
python3 run_eval.py              # plays them into the room as a simulated user
python3 score_eval.py            # scores results + computes latency percentiles

# two callers in the room at once, each running the suite on its own identity
python3 run_eval.py --users 2
python3 score_eval.py            # scores each conversation separately
```

`run_eval.py` writes `.run/eval_manifest.json` — the eval window, which identity played which case, and when. `score_eval.py` reads it, so it never has to guess how to split the trace. Useful flags: `--users N` (concurrent callers), `--cases a,b` (subset), `--rotate N` (start each caller `N` cases into the suite, so the two conversations don't stay in lockstep), `--no-pace` (dump clips as fast as the CPU allows instead of in real time), `--json` (machine-readable report).

### Scoring is per session, and matches on what was heard

Two callers in one room means the agent interleaves two conversations on one CPU, so `call_id`s from different callers land in the trace in whatever order the LLM answered. Scoring therefore runs per `session_id`, and within a session each expected case is matched to the call whose **transcript** contains that case's utterance (`expect_heard` in `score_eval.py`), not to the nth call in the file. That is what keeps the score honest when calls arrive out of order, when a turn is dropped, or when the two callers are running rotated scripts. Order is only the last-resort fallback, and results matched that way are labelled as such. Calls that match no expected case are listed rather than silently consumed.

`python3 test_scoring.py` pins that attribution logic offline, in about a second, on a synthetic interleaved trace — including a dropped turn, a misheard utterance, and a stray call. Latest two-user run: **13/14** across two interleaved conversations (`.run/score_2user.txt`), every case attributed to the right call in the right session; the single failure was the flaky `handoff_trigger` case documented above.

## Observability dashboard

Every call is logged stage-by-stage to a SQLite database, `call_trace.db` (`TRACE_DB` to move it). To view:
```bash
python3 build_dashboard_data.py
python3 -m http.server 8000
# open http://localhost:8000/dashboard.html
```

The trace is four tables rather than a flat append-only file: `sessions` (one per participant connection), `calls` (one per turn — `call_id` is fresh per turn, `session_id` is what holds the conversation together), `events` (one per stage, with the full payload as JSON in `content` and the commonly-queried fields as real columns), and `agent_transitions` (one row per actual change of active agent). Questions the JSONL forced into a re-read-and-regroup are now a single query, and the scorer filters on the events table's epoch column instead of reading the whole trace and dropping what is outside the run.

Traces recorded before the move are still scoreable:
```bash
python3 score_eval.py --import-jsonl call_trace.jsonl   # one-off, idempotent
```

## When a dependency goes down

Every stage is allowed to fail. A turn that cannot be completed still ends with a
spoken sentence and a trace row explaining itself, because the alternative — a
caller who spoke and heard nothing, with no record of why — is indistinguishable
from a broken microphone.

| Stage | Timeout | On failure |
| --- | --- | --- |
| Ollama | `LLM_TIMEOUT_S` (30s), one retry after `LLM_RETRY_BACKOFF_S` | Speaks a short apology, logs `llm_unavailable` |
| MCP tool | `MCP_TOOL_TIMEOUT_S` (10s) | Speaks "I couldn't complete that action", logs `tool_error` with `timed_out` |
| Piper | `PIPER_TIMEOUT_S` (30s) | Skips playback, logs `tts` with `played: false` |
| whisper.cpp | `STT_TIMEOUT_S` (30s) | Ends the turn silently, logs `stt` with the reason |

Anything unforeseen inside a turn is caught at the top of the task, logged with
its traceback as `turn_failed`, and the remaining turns in that session carry on.
A trace-write failure is itself caught: losing observability must not cost a
caller their answer.

These paths are covered offline by `python3 test_degradation.py`, which fails
each dependency in turn against a stubbed pipeline — including a tool that never
returns, and the case where whisper.cpp exits non-zero without writing anything
to stderr, so the trace still records *why* rather than a blank reason.
