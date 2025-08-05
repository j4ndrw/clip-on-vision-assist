import time
from typing import Optional

import pywifi
from src.control_center.models.error import Error
from src.control_center.env import environment


def connect_to_network(
    *,
    target_iface: str = "wlan0",
    ssid: str,
    password: str,
) -> Optional[Error]:
    wifi = pywifi.PyWiFi()
    iface: Optional[pywifi.wifi.Interface] = None
    for _iface in wifi.interfaces():
        if _iface.name() == target_iface:
            iface = _iface

    if iface is None:
        return Error(message=f"Wi-Fi interface `{target_iface}` was not found.")

    profile = pywifi.Profile()
    profile.ssid = ssid # pyright: ignore
    profile.auth = pywifi.const.AUTH_ALG_OPEN
    profile.akm.append(pywifi.const.AKM_TYPE_WPA2PSK)
    profile.cipher = pywifi.const.CIPHER_TYPE_CCMP
    profile.key = password # pyright: ignore


    iface.disconnect()
    iface.connect(iface.add_network_profile(profile))
    time.sleep(10)

    if iface.status() != pywifi.const.IFACE_CONNECTED:
        return Error(message=f"Could not connect to wi-fi `{ssid}`")

    for profile in iface.scan_results():
        if profile.ssid != ssid:
            iface.remove_network_profile(profile)
    environment.update(key="WIFI_SSID", value=ssid).update(key="WIFI_PASSWORD", value=password)
    return None
