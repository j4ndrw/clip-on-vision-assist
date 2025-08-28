import bleak

from src.control_center.models.bluetooth.bluetooth import BluetoothDevice


async def get_bluetooth_devices():
    devices = await bleak.BleakScanner.discover()
    return [
        BluetoothDevice(name=device.name, mac_address=device.address)
        for device in devices
    ]
