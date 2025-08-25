import asyncio
import sys
import json

import httpx

from src.ai_stream_client.api import API
from src.ai_stream_client.client import AIStreamClient
from src.ai_stream_client.state_machines.state import State
from src.ai_stream_client.state_machines.state_machine import StateMachine, StateMachineConfig
from src.ai_stream_client.state_machines.register import currently_active_state_machine
from src.control_center.services.bluetooth.connect_bluetooth_headphones import connect_bluetooth_headphones
from src.env import environment
from src.control_center.services.os.audio.set_audio_device_to_hands_free_mode import set_audio_device_to_hands_free_mode_async
from src.control_center.services.wifi.connect_to_network import connect_to_network_async


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
        try:
            await set_audio_device_to_hands_free_mode_async()
            await asyncio.sleep(5)
        except Exception:
            bluetooth_headphones_mac_address = environment.get()["BLUETOOTH_HEADPHONES_MAC"]
            if bluetooth_headphones_mac_address is None:
                print("Bluetooth headphones must be connected!")
                sys.exit(1)

            err = await connect_bluetooth_headphones(
                mac_address=bluetooth_headphones_mac_address,
                preparation_step=None
            )
            if err is None:
                print("Bluetooth headphones connected!")
                break

            print(f"Bluetooth headphones not connected - {err.message}")
            await asyncio.sleep(1)
        finally:
            await asyncio.sleep(5)
            await set_audio_device_to_hands_free_mode_async()

    while True:
        wifi_ssid = environment.get()["WIFI_SSID"]
        wifi_password = environment.get()["WIFI_PASSWORD"]
        if wifi_ssid is None or wifi_password is None:
            raise Exception("No Wi-Fi credentials supplied. Aborting...")

        print(f"Connecting to SSID `{wifi_ssid}`")
        err = await connect_to_network_async(ssid=wifi_ssid, password=wifi_password)
        if err is None:
            print(f"Connected to wifi `{wifi_ssid}`")
            break


        print(f"Could not connect to wifi - `{err.message}`")
        await asyncio.sleep(5)

    while True:
        client = AIStreamClient()
        state = State()
        state_machine = currently_active_state_machine["machine"]

        while True:
            try:
                llm_backend_endpoint = environment.get()["LLM_BACKEND_ENDPOINT"]
                llm_backend_api_key = environment.get()["LLM_BACKEND_API_KEY"]

                assert llm_backend_endpoint is not None, "`LLM_BACKEND_ENDPOINT` environment variable not set!"
                assert llm_backend_api_key is not None, "`LLM_BACKEND_API_KEY` environment variable not set!"

                API.healthcheck()
                async with httpx.AsyncClient(headers={
                    "x-llm-backend-endpoint": llm_backend_endpoint,
                    "x-llm-backend-api-key": llm_backend_api_key,
                }) as http_client:
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
