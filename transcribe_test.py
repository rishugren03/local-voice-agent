import asyncio
import os
import wave
import subprocess
import numpy as np
from dotenv import load_dotenv
from livekit import rtc, api
import torch

from silero_vad import load_silero_vad

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

        vad_model = load_silero_vad()
        window_size = 512 # required by silero at 16kHz
        rolling_buffer = np.array([], dtype=np.int16)
        speech_buffer = []
        is_speaking = False
        silence_frames = 0
        SILENCE_THRESHOLD = 20 # ~20 windows of silence (~640ms) = end of turn

        async for event in stream:
            frame = event.frame
            data = np.frombuffer(frame.data, dtype=np.int16)
            rolling_buffer = np.concatenate([rolling_buffer, data])

            while len(rolling_buffer) >= window_size:
                chunk = rolling_buffer[:window_size]
                rolling_buffer = rolling_buffer[window_size:]

                float_chunk = chunk.astype(np.float32) / 32768.0
                speech_prob = vad_model(torch.from_numpy(float_chunk), sample_rate).item()

                if speech_prob > 0.5:
                    is_speaking = True
                    silence_frames = 0
                    speech_buffer.append(chunk)

                elif is_speaking:
                    silence_frames += 1
                    speech_buffer.append(chunk)

                    if silence_frames >= SILENCE_THRESHOLD:
                        # End of turn detected
                        full_audio = np.concatenate(speech_buffer)
                        speech_buffer = []
                        is_speaking = False
                        silence_frames = 0
                        print("[DEBUG] End of turn detected, transcribing...")
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
                        