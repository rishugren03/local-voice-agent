"""Two-participant concurrency probe.

Joins N synthetic participants to the agent room simultaneously, has each one
speak a scripted sequence of pre-synthesized utterances in overlapping windows,
and MEASURES the audio each participant actually receives.

The measurement is the point: instead of eyeballing a transcript, every
participant subscribes to every agent track and records per-250ms RMS energy.
That turns "does session A's response leak into session B's ears" into a number
you can diff, and it works without a human in the loop.

Usage:
    python3 concurrency_probe.py --out concurrency_before.json
    python3 concurrency_probe.py --out concurrency_after.json
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
WINDOW_S = 0.25

# (seconds from probe start, wav id). The two scripts interleave on purpose:
# alice's handoff turn lands while bob is mid-conversation on the default agent.
SCENARIOS = {
    "alice": [
        (1.0, "handoff_trigger"),   # primary -> scheduler
        (13.0, "off_topic"),        # scheduler -> primary
        (26.0, "off_topic"),        # primary -> trivia
        (40.0, "off_topic"),        # stays on trivia (alice pinned to amy voice)
        (50.0, "off_topic"),
    ],
    "bob": [
        (4.0, "math_simple"),       # stays on primary
        (17.0, "greeting"),
        (29.0, "math_complex"),     # after alice is on trivia, must still be primary
        (41.0, "greeting"),
        # barge-in: bob talks again 1.2s later, while his own answer is still
        # playing. Must truncate bob's playback only, never alice's.
        (42.2, "math_simple"),
        (54.0, "math_simple"),
    ],
}


async def publish_wav(source: rtc.AudioSource, wav_path: str, pace: bool = True):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16)
    frame_size = 480
    for i in range(0, len(samples), frame_size):
        chunk = samples[i:i + frame_size]
        frame = rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=sr,
            num_channels=1,
            samples_per_channel=len(chunk),
        )
        await source.capture_frame(frame)
        if pace:
            await asyncio.sleep(len(chunk) / sr)


class Participant:
    """A synthetic caller: publishes speech, and meters every agent track it hears."""

    def __init__(self, identity, script):
        self.identity = identity
        self.script = script
        self.room = rtc.Room()
        self.tracks = {}       # track sid -> {"name":..., "windows":[[t, rms], ...]}
        self.spoke = []        # [[t, wav_id], ...]
        self._tasks = []

    def _meter(self, track, publication, owner_identity):
        async def run():
            stream = rtc.AudioStream(track)
            record = {"name": publication.name, "source": str(publication.source),
                      "sid": publication.sid, "identity": owner_identity,
                      "windows": []}
            self.tracks[publication.sid] = record
            # Fixed wall-clock grid, not "flush 250ms after the first frame".
            # A frame-driven grid hides the gaps: if the sender dumps audio into
            # the queue faster than real time, the receiver sees a short burst and
            # the timeline would look continuous when nothing was flowing.
            # Buckets are appended as they fill, so a partial timeline is still
            # valid if the report is written before the track closes.
            buckets = {}
            grid0 = None
            last = None
            flushed_up_to = -1
            async for event in stream:
                pcm = np.frombuffer(event.frame.data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float32)))))
                ts = time.monotonic()
                if grid0 is None:
                    grid0 = ts
                if last is not None and ts - last > 2 * WINDOW_S:
                    # A real gap in delivery: write out what we have, then resync
                    # the grid so the silence stays a hole instead of a bridge.
                    self._flush(buckets, record, grid0)
                    buckets, grid0, flushed_up_to = {}, ts, -1
                last = ts
                idx = int((ts - grid0) / WINDOW_S)
                if idx not in buckets or rms > buckets[idx]:
                    buckets[idx] = rms
                if idx > flushed_up_to + 2:
                    # keep the timeline close to live rather than buffering it all
                    flushed_up_to = idx - 2
                    self._flush(buckets, record, grid0, up_to=flushed_up_to)
            self._flush(buckets, record, grid0)

        self._tasks.append(asyncio.create_task(run()))

    @staticmethod
    def _flush(buckets, record, grid0, up_to=None):
        for idx in sorted(buckets):
            if up_to is not None and idx > up_to:
                break
            record["windows"].append([round(grid0 + (idx + 1) * WINDOW_S, 3),
                                      round(buckets[idx], 1)])
            del buckets[idx]

    async def run(self, start_at):
        token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
            .with_identity(self.identity) \
            .with_name(self.identity) \
            .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME)) \
            .to_jwt()

        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                print(f"[{self.identity}] subscribed to audio track "
                      f"name={publication.name} sid={publication.sid} "
                      f"from={participant.identity}", flush=True)
                self._meter(track, publication, participant.identity)

        await self.room.connect(os.getenv("LIVEKIT_URL"), token)
        print(f"[{self.identity}] joined room as {self.room.name}")

        source = rtc.AudioSource(22050, 1)
        track = rtc.LocalAudioTrack.create_audio_track(f"user-{self.identity}", source)
        await self.room.local_participant.publish_track(track)

        t0 = start_at
        for at, wav_id in self.script:
            delay = at - (time.monotonic() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            wav_path = os.path.join(TEST_AUDIO_DIR, f"{wav_id}.wav")
            print(f"[{self.identity}] speaking: {wav_id} @ t+{time.monotonic() - t0:.1f}s")
            self.spoke.append([round(time.monotonic(), 3), wav_id])
            await publish_wav(source, wav_path)

        end_delay = 62 - (time.monotonic() - t0)
        if end_delay > 0:
            await asyncio.sleep(end_delay)

    def report(self):
        return {
            "identity": self.identity,
            "spoke": self.spoke,
            "tracks": list(self.tracks.values()),
        }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="concurrency_probe.json")
    ap.add_argument("--participants", default="alice,bob")
    args = ap.parse_args()

    names = [n.strip() for n in args.participants.split(",") if n.strip()]
    participants = [Participant(n, SCENARIOS[n]) for n in names]

    # Metered windows are stamped with time.monotonic(); the trace is stamped
    # with wall clock. Capture both origins at the same instant so the analyzer can
    # convert one into the other and line turns up with the audio they produced.
    start_mono = time.monotonic()
    start_wall = time.time()
    await asyncio.gather(*(p.run(start_mono) for p in participants))

    # Let the meters drain the last frames before snapshotting the timeline.
    await asyncio.sleep(2)

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_clock": round(time.monotonic() - start_mono, 1),
        "start_mono": start_mono,
        "start_wall": start_wall,
        "participants": [p.report() for p in participants],
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[PROBE] wrote {args.out}")
    for p in report["participants"]:
        for t in p["tracks"]:
            peak = max((w[1] for w in t["windows"]), default=0)
            print(f"[PROBE] {p['identity']} hears track '{t['name']}' ({t['sid']}, "
                  f"from {t['identity']}): {len(t['windows'])} windows, peak rms={peak}")

    for p in participants:
        await p.room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
