from src.control_center.models.base import BaseSchema
from src.control_center.models.bluetooth.bluetooth import BluetoothDevice


class GetBluetoothDevicesResponse(BaseSchema):
    bluetooth_devices: list[BluetoothDevice]
