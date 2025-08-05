import asyncio
import json

import httpx

from src.ai_stream_client.api import API
from src.ai_stream_client.client import AIStreamClient
from src.ai_stream_client.state_machines.state import State
from src.ai_stream_client.state_machines.state_machine import StateMachine, StateMachineConfig
from src.ai_stream_client.state_machines.register import currently_active_state_machine
from src.control_center.services.bluetooth.connect_bluetooth_headphones import connect_bluetooth_headphones
from src.control_center.env import environment
from src.control_center.services.os.audio.set_audio_device_to_hands_free_mode import set_audio_device_to_hands_free_mode
from src.control_center.services.wifi.connect_to_network import connect_to_network


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
        wifi_ssid = environment.get()["WIFI_SSID"]
        wifi_password = environment.get()["WIFI_PASSWORD"]
        if wifi_ssid is None or wifi_password is None:
            raise Exception("No Wi-Fi credentials supplied. Aborting...")
        err = connect_to_network(ssid=wifi_ssid, password=wifi_password)
        if err is None:
            break

        await asyncio.sleep(10)

    while True:
        bluetooth_headphones_mac_address = environment.get()["BLUETOOTH_HEADPHONES_MAC"]
        if bluetooth_headphones_mac_address is None:
            raise Exception("Bluetooth headphones must be connected!")

        err = await connect_bluetooth_headphones(mac_address=bluetooth_headphones_mac_address)
        if err is None:
            break

        await asyncio.sleep(1)

    await asyncio.sleep(5)
    set_audio_device_to_hands_free_mode()

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
