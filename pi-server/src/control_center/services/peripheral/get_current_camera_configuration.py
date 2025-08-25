from src.control_center.models.peripheral.camera import CameraConfig


def get_current_camera_configuration() -> CameraConfig:
    return CameraConfig.from_environment()
