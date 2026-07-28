import asyncio
import os
import wave
import subprocess
import numpy as np
from dotenv import load_dotenv
from livekit import rtc, api

load_dotenv()

WHISPER_BIN = "/home/rishu/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/rishu/whisper.cpp/models/ggml-base.en.bin"
CHUNK_SECONDS = 4

async def main():
    url = os.getenv('LIVEKIT_URL')
    token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
        .with_identity("voice-agent") \
        .with_name("Voice Agent") \
        .with_grants(api.VideoGrants(room_join=True, room="test-room")) \
        .to_jwt()

    room = rtc.Room()
    audio_buffer = []
    sample_rate = 16000 # whisper wants 216KHz mono

    async def process_track(track: rtc.Track):
        stream = rtc.AudioStream(track, sample_rate=sample_rate, num_channels=1)
        frames_collected = 0
        frames_needed = sample_rate * CHUNK_SECONDS

        async for event in stream:
            frame = event.frame
            data = np.frombuffer(frame.data, dtype=np.int16)
            audio_buffer.append(data)
            print(f"[DEBUG] frames_collected: {frames_collected}/{frames_needed}", end="\r")
            frames_collected += len(data)

            if frames_collected >= frames_needed:
                full_audio = np.concatenate(audio_buffer)
                audio_buffer.clear()
                frames_collected = 0
                await transcribe(full_audio)

    async def transcribe(audio_data):
        wav_path = "chunk.wav"
        with wave.open(wav_path, "wb") as wf:
            wf.setparams((1, 2, sample_rate, 0, "NONE", "NONE"))
            wf.writeframes(audio_data.tobytes())

        result = subprocess.run(
            [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav_path, "-nt"],
            capture_output=True, text=True
        )
        text = result.stdout.strip()
        if text:
            print(f"You said: {text}")
            await get_llm_response(text)

    async def get_llm_response(user_text):
        result = subprocess.run(
            ["ollama", "run", "phi4-mini", user_text],
            capture_output=True, text=True, timeout=30
        )
        response = result.stdout.strip()
        print(f"Agent: {response}")
        await speak(response)

    async def speak(text):
        output_wav = "response.wav"
        result = subprocess.run(
            ["piper", "--model", "/home/rishu/en_US-lessac-medium.onnx", "--output_file", output_wav],
            input=text, text=True, capture_output=True
        )
        print(f"[DEBUG] piper returncode: {result.returncode}")
        print(f"[DEBUG] piper stderr: {result.stderr!r}")

        if result.returncode != 0 or not os.path.exists(output_wav):
            print("[DEBUG] Piper failed to produce output, skipping playback")
            return

        await publish_audio(output_wav)

    async def publish_audio(wav_path):
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())

        source = rtc.AudioSource(sr, 1) # sample rate, mono
        track = rtc.LocalAudioTrack.create_audio_track("agent-voice", source)
        await room.local_participant.publish_track(track)

        samples = np.frombuffer(audio_data, dtype=np.int16)
        frame_size = 480 # 10ms chunks at typical sample rates
        for i in range(0, len(samples), frame_size):
            chunk = samples[i: i+frame_size]
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sr,
                num_channels=1,
                samples_per_channel=len(chunk)
            )
            await source.capture_frame(frame)              

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"Audio track subscribed from {participant.identity}, starting transcription...")
            asyncio.create_task(process_track(track))

    await room.connect(url, token)
    print(f"Agent joined room: {room.name}")
    print("Speak in the other tab — transcripts will print every ~4 seconds.")

    await asyncio.sleep(120)
    await room.disconnect()

asyncio.run(main())
                        