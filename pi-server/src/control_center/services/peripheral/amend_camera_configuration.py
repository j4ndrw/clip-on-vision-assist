from typing import Optional

from src.control_center.models.error import Error
from src.control_center.models.peripheral.camera import CameraConfig


def amend_camera_configuration(*, camera_config: CameraConfig) -> Optional[Error]:
    camera_config.save_to_environment()
