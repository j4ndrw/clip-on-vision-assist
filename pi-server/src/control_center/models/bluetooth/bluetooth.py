from typing import Optional
from src.control_center.models.base import BaseSchema


class BluetoothDevice(BaseSchema):
    name: Optional[str]
    mac_address: str
