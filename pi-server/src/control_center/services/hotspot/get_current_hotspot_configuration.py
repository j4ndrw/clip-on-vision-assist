from src.control_center.models.hotspot.hotspot import HotspotConfig


def get_current_hotspot_configuration() -> HotspotConfig:
    return HotspotConfig.from_environment()
