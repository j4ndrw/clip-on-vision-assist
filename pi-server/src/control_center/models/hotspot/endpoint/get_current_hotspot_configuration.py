from src.control_center.models.base import BaseSchema
from src.control_center.models.hotspot.hotspot import HotspotConfigDTO


class GetCurrentHotspotConfigurationResponse(BaseSchema):
    hotspot_config: HotspotConfigDTO
