from src.control_center.models.base import BaseSchema
from src.control_center.models.peripheral.camera import CameraConfig


class GetCurrentCameraConfigurationResponse(BaseSchema):
    camera_config: CameraConfig
