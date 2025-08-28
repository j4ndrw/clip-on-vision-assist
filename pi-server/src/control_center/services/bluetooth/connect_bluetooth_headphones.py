import asyncio
from typing import Any, Callable, Coroutine, Optional

import bleak

from src.control_center.models.error import Error
from src.env import environment


async def disconnect_from_current_device():
    current_device_mac_address = environment.get()["BLUETOOTH_HEADPHONES_MAC"]
    if current_device_mac_address is not None:
        client = bleak.BleakClient(current_device_mac_address)
        try:
            await client.disconnect()
        except Exception:
            pass


async def connect_bluetooth_headphones(
    *,
    mac_address: str,
    preparation_step: Optional[
        Callable[[], Coroutine[Any, Any, None]]
    ] = disconnect_from_current_device,
) -> Optional[Error]:
    if preparation_step is not None:
        await preparation_step()

    current_device_mac_address = environment.get()["BLUETOOTH_HEADPHONES_MAC"]
    if current_device_mac_address is not None:
        client = bleak.BleakClient(current_device_mac_address)
        try:
            await client.disconnect()
        except Exception:
            pass

    await asyncio.sleep(3)

    client = bleak.BleakClient(mac_address, pair=True)
    try:
        if not client.is_connected:
            await client.connect()
        environment.update(key="BLUETOOTH_HEADPHONES_MAC", value=mac_address)
    except Exception as e:
        print(e)
        return Error(message=f"Could not connect to bluetooth device `{mac_address}`")

    return None
