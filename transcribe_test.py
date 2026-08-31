import asyncio
import os
import wave
import json
import re
import subprocess
import numpy as np
from dotenv import load_dotenv
from livekit import rtc, api
import torch
import requests

from silero_vad import load_silero_vad
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

WHISPER_BIN = "/home/rishu/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/rishu/whisper.cpp/models/ggml-base.en.bin"
CHUNK_SECONDS = 4
AGENT_STATE = {"mode": "LISTENING", "interrupt": False}

AGENTS = {
    "primary": {
        "name": "primary",
        "system": "You are a helpful general assistant. If the user wants to check calendar availability or schedule something, hand off to the scheduler.",
    },
    "scheduler": {
        "name": "scheduler",
        "system": "You are a scheduling assistant. Help the user check availability and find appointment slots using the check_calendar tool. Be efficient and specific.",
    },
}
CURRENT_AGENT = {"active": "primary", "context": ""}

async def main():
    mcp_params = StdioServerParameters(
        command="python3",
        args=["mcp_server.py"],
    )
    mcp_stdio_ctx = stdio_client(mcp_params)
    mcp_read, mcp_write = await mcp_stdio_ctx.__aenter__()
    mcp_session = ClientSession(mcp_read, mcp_write)
    await mcp_session.__aenter__()
    await mcp_session.initialize()
    print("[DEBUG] MCP server connected")

    url = os.getenv('LIVEKIT_URL')
    token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")) \
        .with_identity("voice-agent") \
        .with_name("Voice Agent") \
        .with_grants(api.VideoGrants(room_join=True, room="test-room")) \
        .to_jwt()
    

    room = rtc.Room()
    sample_rate = 16000  # whisper wants 16kHz mono

    # --- Persistent audio source/track for the agent's voice, published once ---
    PIPER_SAMPLE_RATE = 22050
    agent_audio_source = rtc.AudioSource(PIPER_SAMPLE_RATE, 1)
    agent_audio_track = rtc.LocalAudioTrack.create_audio_track("agent-voice", agent_audio_source)

    async def get_llm_response(user_text):
        active = CURRENT_AGENT["active"]
        persona = AGENTS[active]
        context_note = f"\n(Context from handoff: {CURRENT_AGENT['context']})" if CURRENT_AGENT["context"] else ""

        if active == "primary":
            tool_prompt = f"""{persona['system']}

You have access to these tools:
- calculate(expression): evaluates a math expression
- check_calendar(date): checks calendar for a date like '2026-08-15'
- handoff_to_scheduler(reason): transfers the conversation to a scheduling specialist

If the user's request needs one of these, respond with ONLY the matching JSON and nothing else:
{{"tool": "calculate", "args": {{"expression": "..."}}}}
{{"tool": "check_calendar", "args": {{"date": "..."}}}}
{{"tool": "handoff_to_scheduler", "args": {{"reason": "..."}}}}

Otherwise, respond normally and conversationally. Do NOT explain your reasoning or mention tools.

User: {user_text}
Agent:"""
        else:  # scheduler persona
            tool_prompt = f"""{persona['system']}{context_note}

You have access to:
- check_calendar(date): checks calendar for a date like '2026-08-15'
- handoff_to_primary(reason): transfers back to the general assistant if the user's request is no longer about scheduling

If the user's request needs one of these, respond with ONLY the matching JSON and nothing else:
{{"tool": "check_calendar", "args": {{"date": "..."}}}}
{{"tool": "handoff_to_primary", "args": {{"reason": "..."}}}}

Otherwise, respond normally and conversationally, focused on scheduling. Do NOT explain your reasoning or mention tools.

User: {user_text}
Agent:"""

        raw_response = call_ollama(tool_prompt, stop=["\nUser", "User:", "\n---"], max_tokens=150)

        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        tool_used = False

        if json_match:
            try:
                call = json.loads(json_match.group())
                if "tool" in call:
                    tool_used = True
                    tool_name = call["tool"]
                    tool_args = call.get("args", {})
                    print(f"[DEBUG] Calling tool: {tool_name}({tool_args})")

                    if tool_name == "handoff_to_scheduler":
                        CURRENT_AGENT["active"] = "scheduler"
                        CURRENT_AGENT["context"] = tool_args.get("reason", "")
                        response = "Sure, let me connect you with scheduling."
                    elif tool_name == "handoff_to_primary":
                        CURRENT_AGENT["active"] = "primary"
                        CURRENT_AGENT["context"] = ""
                        response = "Sure, let me bring you back to the main assistant."
                    else:
                        tool_result = await mcp_session.call_tool(tool_name, tool_args)
                        result_text = tool_result.content[0].text if tool_result.content else "No result"
                        print(f"[DEBUG] Tool result: {result_text}")

                        response = call_ollama(
                            f"The tool returned: {result_text}. Respond to the user naturally with this information, in one short sentence.\nAgent:",
                            stop=["\nUser", "User:"], max_tokens=60
                        )
            except (json.JSONDecodeError, KeyError):
                tool_used = False

        if not tool_used:
            response = raw_response

        response = re.split(r'\n---\n|\*\*Note:?\*\*|^Note:', response, maxsplit=1)[0].strip()
        print(f"[DEBUG] Active agent: {CURRENT_AGENT['active']}")
        print(f"Agent: {response}")
        await speak(response)
    

    async def process_track(track: rtc.Track):
        stream = rtc.AudioStream(track, sample_rate=sample_rate, num_channels=1)

        vad_model = load_silero_vad()
        window_size = 512  # required by silero at 16kHz
        rolling_buffer = np.array([], dtype=np.int16)
        speech_buffer = []
        is_speaking = False
        silence_frames = 0
        SILENCE_THRESHOLD = 20  # ~20 windows of silence (~640ms) = end of turn

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
                    if AGENT_STATE["mode"] == "SPEAKING":
                        AGENT_STATE["interrupt"] = True
                        print("[DEBUG] Barge-in detected")
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
                        asyncio.create_task(transcribe(full_audio))   # <-- changed from `await transcribe(...)`

    def call_ollama(prompt, stop=None, max_tokens=150):
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi4-mini",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "stop": stop or [],
                    "num_predict": max_tokens
                }
            },
            timeout=30
        )
        return response.json()["response"].strip()

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
        # Uses agent_audio_source from the enclosing scope — no need to pass it in,
        # since this track is published once and reused for every response.
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())

        samples = np.frombuffer(audio_data, dtype=np.int16)
        frame_size = 480
        AGENT_STATE["mode"] = "SPEAKING"
        AGENT_STATE["interrupt"] = False

        for i in range(0, len(samples), frame_size):
            if AGENT_STATE["interrupt"]:
                print("[DEBUG] Playback interrupted by barge-in")
                break
            chunk = samples[i:i + frame_size]
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=sr,
                num_channels=1,
                samples_per_channel=len(chunk)
            )
            await agent_audio_source.capture_frame(frame)

        AGENT_STATE["mode"] = "LISTENING"

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"Audio track subscribed from {participant.identity}, starting transcription...")
            asyncio.create_task(process_track(track))

    await room.connect(url, token)
    print(f"Agent joined room: {room.name}")

    # Publish the agent's voice track once, right after connecting
    await room.local_participant.publish_track(agent_audio_track)
    print("Agent audio track published.")

    print("Speak in the other tab — the agent will respond, and you can interrupt it by talking.")

    await asyncio.sleep(120)
    await room.disconnect()
    await mcp_session.__aexit__(None, None, None)
    await mcp_stdio_ctx.__aexit__(None, None, None)

asyncio.run(main())
