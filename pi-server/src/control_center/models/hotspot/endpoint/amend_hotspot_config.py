from src.control_center.models.base import BaseSchema
from src.control_center.models.hotspot.hotspot import HotspotConfig


class AmendHotspotConfigurationRequest(BaseSchema):
    hotspot_config: HotspotConfig
