"""Deterministic per-session isolation checks — no LiveKit, no Ollama, no Piper.

The live two-participant experiment (concurrency_probe.py) proves isolation end to
end but takes a couple of minutes and depends on every local service being up.
These checks pin the same invariants at the state-machine level in about a second,
so a regression in per-session behaviour fails fast instead of needing a full run.

Run: python3 test_session_isolation.py
"""

import asyncio
import os
import sys
import tempfile
import wave

import numpy as np

import transcribe_test as agent

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")


class FakeSource:
    """Stand-in for rtc.AudioSource. Records what was played and how playback ended,
    so the test can assert WHICH session's queue was touched."""

    def __init__(self, rate=22050, interrupt_after=None):
        self.sample_rate = rate
        self.num_channels = 1
        self.frames = 0
        self.cleared = 0
        self.waited = 0
        self._interrupt_after = interrupt_after
        self._on_capture = None

    async def capture_frame(self, frame):
        self.frames += 1
        if self._on_capture:
            self._on_capture(self)

    def clear_queue(self):
        self.cleared += 1

    async def wait_for_playout(self):
        self.waited += 1


def make_session(session_id, identity, source):
    s = agent.Session(session_id, identity)
    s.source = source
    return s


def write_wav(path, seconds=1.0, rate=22050):
    samples = (np.sin(np.linspace(0, 40 * np.pi, int(rate * seconds))) * 8000).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setparams((1, 2, rate, 0, "NONE", "NONE"))
        wf.writeframes(samples.tobytes())
    return path


async def test_audio_never_crosses_sessions():
    print("\n1. Each session publishes only into its own audio source")
    a_src, b_src = FakeSource(), FakeSource()
    a = make_session("alice-1", "alice", a_src)
    b = make_session("bob-1", "bob", b_src)

    with tempfile.TemporaryDirectory() as d:
        wav = write_wav(os.path.join(d, "turn.wav"))
        ra, rb = await asyncio.gather(
            agent.publish_audio(a, "call-a", wav),
            agent.publish_audio(b, "call-b", wav),
        )

    check("both sessions produced audio", a_src.frames > 0 and b_src.frames > 0,
          f"alice={a_src.frames} frames, bob={b_src.frames} frames")
    check("playback completed for both", ra["end_reason"] == "completed" and rb["end_reason"] == "completed",
          f"alice={ra['end_reason']}, bob={rb['end_reason']}")
    check("neither session cleared the other's queue", a_src.cleared == 0 and b_src.cleared == 0)
    check("each session has its own track object", a.track is not b.track,
          f"{a.track.name} vs {b.track.name}")


async def test_barge_in_is_per_session():
    print("\n2. A barge-in truncates only the interrupting session's playback")
    a_src = FakeSource()
    b_src = FakeSource()
    a = make_session("alice-2", "alice", a_src)
    b = make_session("bob-2", "bob", b_src)

    # Alice's participant starts talking again 20 frames into her own playback.
    # This is the exact scenario the old global AGENT_STATE dict broke: the flag
    # was shared, so Alice talking could cut Bob off mid-sentence.
    def alice_speaks(source):
        if source.frames == 20:
            a.playback.interrupt = True

    a_src._on_capture = alice_speaks

    with tempfile.TemporaryDirectory() as d:
        wav = write_wav(os.path.join(d, "turn.wav"), seconds=2.0)
        ra, rb = await asyncio.gather(
            agent.publish_audio(a, "call-a", wav),
            agent.publish_audio(b, "call-b", wav),
        )

    check("alice's playback was interrupted", ra["end_reason"] == "barge_in", ra["end_reason"])
    check("bob's playback was NOT interrupted", rb["end_reason"] == "completed", rb["end_reason"])
    check("only alice's queue was cleared", a_src.cleared == 1 and b_src.cleared == 0,
          f"alice cleared={a_src.cleared}, bob cleared={b_src.cleared}")
    check("bob heard his full turn while alice was cut off",
          b_src.frames > a_src.frames, f"bob={b_src.frames} frames vs alice={a_src.frames}")
    check("alice's playback state returned to LISTENING", a.playback.mode == "LISTENING")


async def test_handoff_state_is_per_session():
    print("\n3. A handoff moves one session's active agent only")
    agent.SESSIONS.clear()
    agent.SESSION_BY_IDENTITY.clear()

    a = agent.get_session("alice-3", "alice")
    b = agent.get_session("bob-3", "bob")
    check("both sessions start on the entry agent", a.active == b.active == "primary")

    # Simulate the handoff branch of get_llm_response for alice only.
    target = agent.REGISTRY.resolve_handoff("handoff_to_trivia") if agent.REGISTRY else "trivia"
    a.active = target
    a.context = "user wants a fun fact"

    check("alice moved to the new agent", a.active == "trivia", a.active)
    check("bob's active agent is untouched", b.active == "primary", b.active)
    check("bob's handoff context is untouched", b.context == "", repr(b.context))

    # Same identity reattaching (mic unmute) must keep its agent, not reset.
    a2 = agent.get_session("alice-3-newid", "alice")
    check("reconnecting identity keeps its active agent", a2 is a and a2.active == "trivia",
          f"same object={a2 is a}, agent={a2.active}")

    other = agent.get_session("carol-3", "carol")
    check("a different participant gets a different session", other is not a)

    agent.SESSIONS.clear()
    agent.SESSION_BY_IDENTITY.clear()


async def test_turn_lock_is_per_session():
    print("\n4. Turns serialize within a session but not across sessions")
    a_src, b_src = FakeSource(), FakeSource()
    a = make_session("alice-4", "alice", a_src)
    b = make_session("bob-4", "bob", b_src)
    # handle_turn ignores turns for calls that have already ended, so the
    # sessions have to be live for their turns to be processed at all.
    agent.SESSIONS[a.id] = a
    agent.SESSIONS[b.id] = b

    active = {"alice": 0, "bob": 0}
    max_active = {"alice": 0, "bob": 0}
    order = []

    async def fake_transcribe(session, audio, call_id):
        who = session.identity
        active[who] += 1
        max_active[who] = max(max_active[who], active[who])
        order.append(f"{who}:{call_id}:start")
        await asyncio.sleep(0.05)
        active[who] -= 1
        order.append(f"{who}:{call_id}:end")

    original = agent.transcribe
    agent.transcribe = fake_transcribe
    try:
        await asyncio.gather(
            agent.handle_turn(a, None, "a1"),
            agent.handle_turn(a, None, "a2"),
            agent.handle_turn(b, None, "b1"),
        )
    finally:
        agent.transcribe = original
        agent.SESSIONS.clear()

    check("alice's two turns never overlapped", max_active["alice"] == 1,
          f"max concurrent={max_active['alice']}")
    check("bob's turn ran at the same time as alice's", max_active["bob"] == 1 and
          order.index("alice:a1:start") < order.index("bob:b1:end"),
          " -> ".join(order))


async def main():
    # Playback pacing is disabled so the checks run at CPU speed; the interrupt
    # flag is still evaluated once per frame, which is what is under test.
    agent.PLAYBACK_LEAD_S = 1e9

    print("Per-session isolation checks")
    await test_audio_never_crosses_sessions()
    await test_barge_in_is_per_session()
    await test_handoff_state_is_per_session()
    await test_turn_lock_is_per_session()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
