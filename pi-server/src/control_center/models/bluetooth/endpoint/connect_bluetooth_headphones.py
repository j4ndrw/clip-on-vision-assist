from src.control_center.models.base import BaseSchema


class ConnectBluetoothHeadphonesRequest(BaseSchema):
    mac_address: str
