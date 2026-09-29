"""Simulated user(s) for the eval suite.

One caller by default. `--users 2` puts two synthetic callers in the same room at
the same time, each running the whole suite on its own LiveKit identity, so the
agent interleaves two real conversations on one CPU. That is what exercises the
scorer's per-session attribution: with both conversations in the trace at
once, a scorer that reads call order globally hands one user's math case the
other user's weather answer.

`--rotate` starts each user's script at a different point in the suite, so the two
conversations do not stay in lockstep. A run with rotation is the sharp version of
the stress test: even within a single session, the case order in the trace no
longer matches the suite's order.

Writes .run/eval_manifest.json — the eval window plus which identity ran which
cases in which order — which score_eval.py reads. Without it the scorer has to
guess the case list from wall-clock windows alone.

Usage:
    python3 run_eval.py
    python3 run_eval.py --users 2 --rotate 3
    python3 run_eval.py --cases math_simple,greeting
"""

import argparse
import asyncio
import json
import os
import time
import wave

import numpy as np
from dotenv import load_dotenv
from livekit import rtc, api

load_dotenv()

ROOM_NAME = os.getenv("ROOM_NAME", "test-room")
TEST_AUDIO_DIR = "test_audio"
MANIFEST_FILE = os.path.join(".run", "eval_manifest.json")

# Piper's actual output sample rate — check with:
# python3 -c "import wave; print(wave.open('test_audio/greeting.wav').getframerate())"
TEST_SAMPLE_RATE = 22050

TEST_CASES = [
    {"id": "greeting", "wait_after_s": 8},
    {"id": "math_simple", "wait_after_s": 8},
    {"id": "math_complex", "wait_after_s": 8},
    {"id": "calendar_check", "wait_after_s": 8},
    {"id": "weather_check", "wait_after_s": 8},
    {"id": "handoff_trigger", "wait_after_s": 8},
    {"id": "off_topic", "wait_after_s": 8},
]


async def publish_wav(source: rtc.AudioSource, wav_path: str, pace: bool = True):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    samples = np.frombuffer(audio_data, dtype=np.int16)
    frame_size = 480
    frame_duration = frame_size / sr
    n_frames = (len(samples) + frame_size - 1) // frame_size

    # Pace against an absolute schedule, not by sleeping frame_duration after each
    # frame. A per-frame sleep accumulates every scheduling delay, so on a loaded
    # machine (two callers plus whisper, piper and ollama on one CPU) the clip is
    # delivered progressively later — the caller then talks over the agent's answer
    # and the tail of the clip is heard as a barge-in. Absolute deadlines mean a
    # late frame only delays itself.
    started_at = time.monotonic()
    for i in range(n_frames):
        chunk = samples[i * frame_size:(i + 1) * frame_size]
        frame = rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=sr,
            num_channels=1,
            samples_per_channel=len(chunk)
        )
        await source.capture_frame(frame)
        if pace:
            delay = started_at + (i + 1) * frame_duration - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    return {"started_at": started_at, "ended_at": time.monotonic()}


class SimUser:
    """One synthetic caller: its own LiveKit identity, audio source and track.

    Separate identities are what give the agent separate sessions; two callers
    sharing an identity would be treated as one session that republished its mic,
    which is the mute/unmute path, not concurrency.
    """

    def __init__(self, label, cases, pace=True):
        self.label = label
        self.identity = f"eval-runner-{label}"
        self.cases = cases
        self.pace = pace
        self.room = rtc.Room()
        self.source = rtc.AudioSource(TEST_SAMPLE_RATE, 1)
        self.track = rtc.LocalAudioTrack.create_audio_track(f"user-{self.label}-voice", self.source)
        self.started_at = None
        self.ended_at = None
        self.played = []

    async def connect(self):
        token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
            .with_identity(self.identity) \
            .with_name(f"Eval Runner {self.label}") \
            .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME)) \
            .to_jwt()
        await self.room.connect(os.getenv("LIVEKIT_URL"), token)
        await self.room.local_participant.publish_track(self.track)
        print(f"[EVAL] {self.label}: joined room as '{self.identity}', published mic")

    async def run_suite(self, start_at):
        """Play this user's cases, aligning to the shared start so the two callers
        genuinely overlap instead of each waiting for the other to finish."""
        self.started_at = time.time()
        for case in self.cases:
            delay = start_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

            wav_path = os.path.join(TEST_AUDIO_DIR, f"{case['id']}.wav")
            if not os.path.exists(wav_path):
                print(f"[EVAL] {self.label}: SKIP {case['id']} (no {wav_path})")
                continue
            print(f"[EVAL] {self.label}: --- {case['id']} ---", flush=True)
            # Epoch timestamps, not just relative ones: these are what the trace's
            # events get compared against to see how far behind the agent the audio
            # actually landed, which is how a late clip shows up as a barge-in.
            played_at = time.time()
            sent = await publish_wav(self.source, wav_path, pace=self.pace)
            self.played.append({
                "case": case["id"],
                "played_at": played_at,
                "clip_s": round(sent["ended_at"] - sent["started_at"], 2),
            })
            print(f"[EVAL] {self.label}: played {case['id']} "
                  f"({self.played[-1]['clip_s']}s of audio at {played_at}), waiting "
                  f"{case['wait_after_s']}s for the answer...", flush=True)
            await asyncio.sleep(case["wait_after_s"])
        self.ended_at = time.time()

    async def disconnect(self):
        await self.room.disconnect()


def rotate(cases, offset):
    """The same suite, started at a different case, wrapping around."""
    if not offset or len(cases) < 2:
        return list(cases)
    offset %= len(cases)
    return list(cases[offset:]) + list(cases[:offset])


async def main():
    ap = argparse.ArgumentParser(description="Run the eval suite as one or more simulated users")
    ap.add_argument("--users", type=int, default=1,
                    help="how many simulated callers to put in the room at once")
    ap.add_argument("--cases", help="comma-separated subset of case ids (default: all)")
    ap.add_argument("--rotate", type=int, default=0,
                    help="start each user's script this many cases into the suite")
    ap.add_argument("--stagger", type=float, default=2.0,
                    help="seconds between each user starting, so the scripts interleave")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds to wait after joining before speaking")
    ap.add_argument("--drain", type=float, default=12.0,
                    help="seconds to stay connected after the last case, so its answer "
                         "finishes and is traced instead of being cut off by the "
                         "disconnect")
    ap.add_argument("--no-pace", dest="pace", action="store_false",
                    help="dump each clip into the audio queue as fast as possible")
    ap.add_argument("--manifest", default=MANIFEST_FILE)
    args = ap.parse_args()

    if args.users < 1:
        raise SystemExit("--users must be at least 1")

    selected = {c["id"]: c for c in TEST_CASES}
    if args.cases:
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in selected]
        if unknown:
            raise SystemExit(f"unknown case(s): {', '.join(unknown)}. "
                             f"Available: {', '.join(selected)}")
        suite = [selected[c] for c in wanted]
    else:
        suite = list(TEST_CASES)

    users = [
        SimUser(label, rotate(suite, i * args.rotate), pace=args.pace)
        for i, label in enumerate("abcdefgh"[:args.users])
    ]

    # Everyone joins first, then waits out the settle, so the agent sees all the
    # tracks in one go rather than attaching sessions one at a time.
    await asyncio.gather(*(u.connect() for u in users))
    print(f"\n[EVAL] {len(users)} user(s) joined; waiting {args.settle}s for the agent "
          f"to notice the new tracks...")
    await asyncio.sleep(args.settle)

    eval_start = time.time()
    start_at = time.monotonic()
    print(f"[EVAL] Starting test suite at {eval_start}")

    await asyncio.gather(*(
        u.run_suite(start_at + i * args.stagger)
        for i, u in enumerate(users)
    ))

    # The last case's answer is still being generated, and the disconnect would
    # cancel it — leaving a turn in the trace with an stt event and no llm event,
    # which the scorer has to report as a failure that never really happened. The
    # window therefore closes after the drain, not before it.
    if args.drain > 0:
        print(f"[EVAL] Draining {args.drain}s so the last answer can finish...")
        await asyncio.sleep(args.drain)

    eval_end = time.time()
    print(f"\n[EVAL] Test suite complete. Window: {eval_start} to {eval_end}")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "room": ROOM_NAME,
        "window": {"start": eval_start, "end": eval_end},
        "cases": [c["id"] for c in suite],
        "paced": args.pace,
        "users": [
            {
                "label": u.label,
                "identity": u.identity,
                "cases": [c["id"] for c in u.cases],
                "started_at": u.started_at,
                "ended_at": u.ended_at,
                "played": u.played,
            }
            for u in users
        ],
    }
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[EVAL] Wrote {args.manifest}")

    print("[EVAL] Score this run with:")
    print(f"[EVAL]   python3 score_eval.py --manifest {args.manifest}")

    await asyncio.gather(*(u.disconnect() for u in users))


if __name__ == "__main__":
    asyncio.run(main())
