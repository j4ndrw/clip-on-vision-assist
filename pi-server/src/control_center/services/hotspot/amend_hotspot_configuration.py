from typing import Optional

from src.control_center.models.error import Error
from src.control_center.models.hotspot.hotspot import HotspotConfig


def amend_hotspot_configuration(
    *, hotspot_config: HotspotConfig
) -> Optional[Error]:
    hotspot_config.save_to_environment()
