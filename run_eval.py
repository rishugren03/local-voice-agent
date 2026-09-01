import asyncio
import os
import time
import wave
import numpy as np
from dotenv import load_dotenv
from livekit import rtc, api

load_dotenv()

TEST_CASES = [
    {"id": "greeting", "wait_after_s": 8},
    {"id": "math_simple", "wait_after_s": 8},
    {"id": "math_complex", "wait_after_s": 8},
    {"id": "calendar_check", "wait_after_s": 8},
    {"id": "handoff_trigger", "wait_after_s": 8},
    {"id": "off_topic", "wait_after_s": 8},
]

TEST_AUDIO_DIR = "test_audio"


async def publish_wav(source: rtc.AudioSource, wav_path: str):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    samples = np.frombuffer(audio_data, dtype=np.int16)
    frame_size = 480
    for i in range(0, len(samples), frame_size):
        chunk = samples[i:i + frame_size]
        frame = rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=sr,
            num_channels=1,
            samples_per_channel=len(chunk)
        )
        await source.capture_frame(frame)


async def main():
    url = os.getenv("LIVEKIT_URL")
    token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
        .with_identity("eval-runner") \
        .with_name("Eval Runner") \
        .with_grants(api.VideoGrants(room_join=True, room="test-room")) \
        .to_jwt()

    room = rtc.Room()
    await room.connect(url, token)
    print(f"[EVAL] Connected to room as 'eval-runner'")

    # Piper's actual output sample rate — check with:
    # python3 -c "import wave; print(wave.open('test_audio/greeting.wav').getframerate())"
    TEST_SAMPLE_RATE = 22050
    source = rtc.AudioSource(TEST_SAMPLE_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("eval-user-voice", source)
    await room.local_participant.publish_track(track)
    print("[EVAL] Published simulated user audio track")

    print("[EVAL] Waiting 3s for agent to notice the new track...")
    await asyncio.sleep(3)

    eval_start_time = time.time()
    print(f"[EVAL] Starting test suite at {eval_start_time}")

    for case in TEST_CASES:
        wav_path = os.path.join(TEST_AUDIO_DIR, f"{case['id']}.wav")
        print(f"\n[EVAL] --- Running case: {case['id']} ---")
        t0 = time.time()
        await publish_wav(source, wav_path)
        print(f"[EVAL] Finished playing {case['id']}, waiting {case['wait_after_s']}s for agent response...")
        await asyncio.sleep(case["wait_after_s"])

    eval_end_time = time.time()
    print(f"\n[EVAL] Test suite complete. Window: {eval_start_time} to {eval_end_time}")
    print(f"[EVAL] Use this window to filter call_trace.jsonl for this run's results.")

    await room.disconnect()


asyncio.run(main())