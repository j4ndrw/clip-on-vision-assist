from src.control_center.models.base import BaseSchema
from src.control_center.models.wifi.wifi import WiFiNetwork


class ScanNetworksResponse(BaseSchema):
    wifi_networks: list[WiFiNetwork]
