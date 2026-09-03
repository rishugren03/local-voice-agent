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

**Tool calls go through MCP, not custom function-calling glue.** Since Phi-4-mini has no native function-calling support (unlike GPT-4/Claude), tool invocation is done via a structured JSON-in-prompt convention, parsed and dispatched to a real MCP server. This keeps the tool layer swappable and standards-based rather than model-specific.

**Small local models need explicit, capitalized rules, not soft suggestions.** During eval testing, Phi-4-mini inconsistently used the `calculate` tool for simple arithmetic — sometimes computing (and getting wrong) answers itself. Softly worded prompts ("if the request needs a tool...") weren't reliable. An explicit rule ("RULE: For ANY math question... you MUST use the calculate tool") fixed this, at the cost of a transient over-triggering regression on unrelated queries during tuning — a real small-model prompt-engineering tradeoff, not a clean fix.

## Reliability data (eval harness)

A scripted 6-case test suite (greeting, simple math, complex math, calendar tool lookup, multi-agent handoff trigger, off-topic robustness) run against a simulated user (pre-synthesized audio injected into the LiveKit room), scored automatically against expected tool usage and response content.

| Metric | Before prompt fix | After prompt fix |
|---|---|---|
| Task pass rate | 4/6 | **6/6** |
| Known failure modes | Unreliable math tool-triggering, handoff not firing | None observed in this run |

**Latency (fully local, single consumer machine, no GPU-specific optimization):**
| Stage | p50 | p90 |
|---|---|---|
| STT (whisper.cpp) | 0.87s | 0.89s |
| LLM (Phi-4-mini) | 2.3s | 2.84s |
| TTS (Piper) | 1.2s | 1.37s |
| **Total time-to-response** | **4.53s** | **4.85s** |

These numbers are from a small eval batch (6 cases) on unoptimized consumer hardware — they represent a baseline, not a ceiling. The LLM stage is the dominant cost; further optimization (smaller quantization, speculative decoding, or streaming token-by-token into TTS) is the clearest path to reducing total latency.

## What's not solved yet

- Latency (~4.5s p50) is well above production voice AI targets (sub-1s is standard for commercial platforms) — this is a known tradeoff of full local inference on consumer hardware, not yet optimized
- No concurrent call handling — single-room, single-conversation design
- Minimal error handling for upstream failures (Ollama crash, Piper failure mid-call)
- STT accuracy on short/noisy utterances is limited by `whisper.cpp base.en` — a larger model would improve this at a latency cost

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

## Eval harness

```bash
python3 generate_test_audio.py   # creates synthetic test utterances
python3 run_eval.py              # plays them into the room as a simulated user
python3 score_eval.py            # scores results + computes latency percentiles
```

## Observability dashboard

Every call is logged stage-by-stage to `call_trace.jsonl`. To view:
```bash
python3 build_dashboard_data.py
python3 -m http.server 8000
# open http://localhost:8000/dashboard.html
```