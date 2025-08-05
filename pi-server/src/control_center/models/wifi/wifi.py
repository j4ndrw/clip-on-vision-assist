from src.control_center.models.base import BaseSchema


class WiFiNetwork(BaseSchema):
    ssid: str
    signal_strength_dbm: int
