import asyncio
import json

import httpx

from src.ai_stream_client.api import API
from src.ai_stream_client.client import AIStreamClient
from src.ai_stream_client.event_loop.utils import (try_connect_audio,
                                                   try_connect_wifi)
from src.ai_stream_client.state_machines.register import \
    currently_active_state_machine
from src.ai_stream_client.state_machines.state import State
from src.ai_stream_client.state_machines.state_machine import (
    StateMachine, StateMachineConfig)
from src.env import environment
from src.utils.decorators import LoopBreak, looped_async, with_interrupt_async


async def receive_event(
    line: str,
    client: AIStreamClient,
    state: State,
    state_machine: StateMachine,
) -> State:
    msg = json.loads(line)
    return state_machine(StateMachineConfig(client=client, state=state, msg=msg))


async def consume_event(state: State):
    if not state.type or not state.task:
        return

    state.task()


def headers():
    llm_backend_endpoint = environment.get()["LLM_BACKEND_ENDPOINT"]
    llm_backend_api_key = environment.get()["LLM_BACKEND_API_KEY"]
    llm = environment.get()["LLM"]

    assert (
        llm_backend_endpoint is not None
    ), "`LLM_BACKEND_ENDPOINT` environment variable not set!"
    assert (
        llm_backend_api_key is not None
    ), "`LLM_BACKEND_API_KEY` environment variable not set!"
    assert llm is not None, "`LLM` environment variable not set!"

    return {
        "x-llm-backend-endpoint": llm_backend_endpoint or "",
        "x-llm-backend-api-key": llm_backend_api_key or "",
        "x-llm": llm or "",
    }


async def process_stream_event(
    *,
    event: str,
    http_client: httpx.AsyncClient,
    ai_stream_client: AIStreamClient,
    state: State,
    state_machine: StateMachine
):
    @with_interrupt_async()
    async def fn(
        *,
        event: str,
        http_client: httpx.AsyncClient,
        ai_stream_client: AIStreamClient,
        state: State,
        state_machine: StateMachine
    ):
        http_client.headers = headers()
        state, _ = await asyncio.gather(
            receive_event(event, ai_stream_client, state, state_machine),
            consume_event(state),
        )

    await fn(
        event=event,
        http_client=http_client,
        ai_stream_client=ai_stream_client,
        state=state,
        state_machine=state_machine,
    )


async def read_stream(
    *, client: AIStreamClient, state: State, state_machine: StateMachine
):
    @with_interrupt_async()
    async def fn(*, client: AIStreamClient, state: State, state_machine: StateMachine):
        async with httpx.AsyncClient(headers=headers()) as http_client:
            async with API.ai_stream(async_http_client=http_client) as ai_stream:
                async for event in ai_stream.aiter_lines():
                    await process_stream_event(
                        event=event,
                        http_client=http_client,
                        ai_stream_client=client,
                        state=state,
                        state_machine=state_machine,
                    )

    await fn(client=client, state=state, state_machine=state_machine)


async def run_state_machine(
    *, client: AIStreamClient, state: State, state_machine: StateMachine
):
    @looped_async
    @with_interrupt_async()
    async def fn(*, client: AIStreamClient, state: State, state_machine: StateMachine) -> LoopBreak:
        try:
            API.healthcheck()
            await read_stream(client=client, state=state, state_machine=state_machine)
            return LoopBreak()
        except httpx.ConnectError:
            await asyncio.sleep(1)
        except httpx.ConnectTimeout:
            await asyncio.sleep(1)

        return LoopBreak()

    await fn(client=client, state=state, state_machine=state_machine)


async def loop():
    @looped_async
    @with_interrupt_async()
    async def fn():
        client = AIStreamClient()
        state = State()
        state_machine = currently_active_state_machine["machine"]

        await run_state_machine(client=client, state=state, state_machine=state_machine)

    await fn()


async def event_loop():
    @with_interrupt_async()
    async def fn():
        await try_connect_audio()
        await try_connect_wifi()
        await loop()

    await fn()
