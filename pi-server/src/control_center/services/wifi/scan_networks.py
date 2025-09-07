import time
from typing import Optional

import pywifi

from src.control_center.models.error import Error
from src.control_center.models.wifi.wifi import WiFiNetwork


def scan_networks(
    *, target_iface: str = "wlan0"
) -> tuple[list[WiFiNetwork], Optional[Error]]:
    wifi = pywifi.PyWiFi()
    iface: Optional[pywifi.wifi.Interface] = None
    for _iface in wifi.interfaces():
        if _iface.name() == target_iface:
            iface = _iface

    if iface is None:
        return [], Error(message=f"Wi-Fi interface `{target_iface}` was not found.")

    iface.scan()
    time.sleep(2)

    wifi_networks = [
        WiFiNetwork(ssid=profile.ssid, signal_strength_dbm=profile.signal)  # pyright: ignore
        for profile in iface.scan_results()
        if isinstance(profile, pywifi.Profile) and profile.ssid is not None
    ]
    return wifi_networks, None
