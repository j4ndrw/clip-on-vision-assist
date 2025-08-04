import asyncio
from typing import Optional
import bleak
from src.control_center.env import environment
from src.control_center.models.error import Error


async def connect_bluetooth_headphones(*, mac_address: str) -> Optional[Error]:
    current_device_mac_address = environment.get()["BLUETOOTH_HEADPHONES_MAC"]
    if current_device_mac_address is not None:
        client = bleak.BleakClient(current_device_mac_address)
        try:
            await client.disconnect()
        except Exception:
            pass

    await asyncio.sleep(1)

    client = bleak.BleakClient(mac_address, pair=True)
    try:
        await client.connect()
        environment.update(key="BLUETOOTH_HEADPHONES_MAC", value=mac_address)
    except Exception as e:
        print(e)
        return Error(message=f"Could not connect to bluetooth device `{mac_address}`")

    return None
