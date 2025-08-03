import asyncio
import json

import httpx

from src.ai_stream_client.api import API
from src.ai_stream_client.client import AIStreamClient
from src.ai_stream_client.state_machines.state import State
from src.ai_stream_client.state_machines.state_machine import StateMachine, StateMachineConfig
from src.ai_stream_client.state_machines.register import currently_active_state_machine


async def receive_event(
    line: str,
    client: AIStreamClient,
    state: State,
    state_machine: StateMachine,
) -> State:
    msg = json.loads(line)
    return state_machine(StateMachineConfig(
        client=client,
        state=state,
        msg=msg
    ))


async def consume_event(state: State):
    if not state.type or not state.task:
        return

    state.task()


async def event_loop():
    while True:
        client = AIStreamClient()
        state = State()
        state_machine = currently_active_state_machine["machine"]

        while True:
            try:
                API.healthcheck()
                async with httpx.AsyncClient() as http_client:
                    async with API.ai_stream(async_http_client=http_client) as ai_stream:
                        async for line in ai_stream.aiter_lines():
                            state, _ = await asyncio.gather(
                                receive_event(line, client, state, state_machine),
                                consume_event(state),
                            )
                break
            except httpx.ConnectError:
                await asyncio.sleep(1)
            except httpx.ConnectTimeout:
                await asyncio.sleep(1)
