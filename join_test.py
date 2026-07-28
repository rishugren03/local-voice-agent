import asyncio
import os
from dotenv import load_dotenv
from livekit import rtc, api

load_dotenv()

async def main():
    url = os.getenv('LIVEKIT_URL')
    token = api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv('LIVEKIT_API_SECRET')) \
       .with_identity("voice-agent") \
       .with_name("Voice Agent") \
       .with_grants(api.VideoGrants(room_join=True, room="test-room")) \
       .to_jwt()

    room = rtc.Room()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        print(f"Subscribed to track: {track.kind} from {participant.identity}")

    await room.connect(url, token)
    print(f"Agent joined room {room.name}")
    print("Waiting for audio tracks... (join test-room from meet.livekit.io in another tab)")

    await asyncio.sleep(60)
    await room.disconnect()

asyncio.run(main())

