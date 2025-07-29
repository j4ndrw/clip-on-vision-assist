import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

import httpx

from src.ai_stream_client.api import API
from src.ai_stream_client.client import AIStreamClient
from src.ai_stream_client.event import StreamEventType
from src.ai_stream_client.tasks import AIStreamTasks


@dataclass
class State:
    type: StreamEventType | None = field(default=None)
    task: Callable[[], None] | None = field(default=None)


async def receive_event(
    ai_stream: AsyncIterator[str], client: AIStreamClient, state: State
) -> State:
    msg = json.loads(await ai_stream.__anext__())
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

    return state


async def consume_event(state: State) -> None:
    if not state.type or not state.task:
        return

    state.task()


async def event_loop():
    client = AIStreamClient()
    state = State()

    async with httpx.AsyncClient() as http_client:
        async with API.ai_stream(async_http_client=http_client) as ai_stream:
            iterator = ai_stream.aiter_lines()
            while True:
                print(state.type)
                state, _ = await asyncio.gather(
                    receive_event(iterator, client, state),
                    consume_event(state),
                )
