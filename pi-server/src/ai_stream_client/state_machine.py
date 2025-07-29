import asyncio
import json

import httpx

from src.ai_stream_client.api import API
from src.ai_stream_client.client import AIStreamClient
from src.ai_stream_client.event import StreamEventType
from src.ai_stream_client.state import State
from src.ai_stream_client.tasks import AIStreamTasks


async def receive_event(line: str, client: AIStreamClient, state: State) -> State:
    msg = json.loads(line)
    state.type = StreamEventType(msg["type"])
    tasks = AIStreamTasks(client=client)

    match state.type:
        case StreamEventType.CAPTURE_WAKEWORD:
            state.task = tasks.capture_wakeword()
        case StreamEventType.CAPTURE_PROMPT:
            state.task = tasks.capture_prompt()
        case StreamEventType.STALL:
            state.task = tasks.stall()
        case StreamEventType.AI_SPEECH:
            state.task = tasks.ai_speech(msg)
        case StreamEventType.DONE:
            state.task = tasks.done()

    return state


async def consume_event(state: State):
    if not state.type or not state.task:
        return False

    state.task()


async def event_loop():
    while True:
        client = AIStreamClient()
        state = State()

        async with httpx.AsyncClient() as http_client:
            async with API.ai_stream(async_http_client=http_client) as ai_stream:
                async for line in ai_stream.aiter_lines():
                    state, _ = await asyncio.gather(
                        receive_event(line, client, state),
                        consume_event(state),
                    )
