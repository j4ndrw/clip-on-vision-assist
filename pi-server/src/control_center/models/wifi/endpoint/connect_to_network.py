from src.control_center.models.base import BaseSchema


class ConnectToNetworkRequest(BaseSchema):
    ssid: str
    password: str
