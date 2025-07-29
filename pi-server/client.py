import asyncio

from src.ai_stream_client.state_machine import event_loop


async def main():
    await event_loop()


if __name__ == "__main__":
    asyncio.run(main())
