from src.control_center.models.base import BaseSchema
from src.control_center.models.peripheral.camera import CameraConfig

class AmendCameraConfigurationRequest(BaseSchema):
    camera_config: CameraConfig
