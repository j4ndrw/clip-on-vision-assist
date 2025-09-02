import asyncio
import sys

from src.control_center.services.bluetooth.connect_bluetooth_headphones import \
    connect_bluetooth_headphones
from src.control_center.services.os.audio.set_audio_device_to_hands_free_mode import \
    set_audio_device_to_hands_free_mode_async
from src.control_center.services.wifi.connect_to_network import \
    connect_to_network_async
from src.env import environment
from src.utils.decorators import (LoopBreak, looped_async,
                                  production_only_async, with_interrupt_async)


async def try_connect_audio():
    @production_only_async
    @looped_async
    @with_interrupt_async()
    async def fn() -> LoopBreak:
        try:
            await set_audio_device_to_hands_free_mode_async()
            await asyncio.sleep(5)
        except Exception:
            bluetooth_headphones_mac_address = environment.get()[
                "BLUETOOTH_HEADPHONES_MAC"
            ]
            if bluetooth_headphones_mac_address is None:
                print("Bluetooth headphones must be connected!")
                sys.exit(1)

            err = await connect_bluetooth_headphones(
                mac_address=bluetooth_headphones_mac_address, preparation_step=None
            )
            if err is None:
                print("Bluetooth headphones connected!")
                return LoopBreak()

            print(f"Bluetooth headphones not connected - {err.message}")
            await asyncio.sleep(1)
        finally:
            await asyncio.sleep(5)
            await set_audio_device_to_hands_free_mode_async()

        return LoopBreak()

    await fn()


async def try_connect_wifi():
    @production_only_async
    @looped_async
    @with_interrupt_async()
    async def fn() -> LoopBreak:
        wifi_ssid = environment.get()["WIFI_SSID"]
        wifi_password = environment.get()["WIFI_PASSWORD"]
        if wifi_ssid is None or wifi_password is None:
            raise Exception("No Wi-Fi credentials supplied. Aborting...")

        print(f"Connecting to SSID `{wifi_ssid}`")
        err = await connect_to_network_async(ssid=wifi_ssid, password=wifi_password)
        if err is None:
            print(f"Connected to wifi `{wifi_ssid}`")
            return LoopBreak()

        print(f"Could not connect to wifi - `{err.message}`")
        await asyncio.sleep(5)
        return LoopBreak()

    await fn()
