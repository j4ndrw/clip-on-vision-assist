import asyncio

from src.ai_stream_client.event_loop import event_loop


if __name__ == "__main__":
    asyncio.run(event_loop())
