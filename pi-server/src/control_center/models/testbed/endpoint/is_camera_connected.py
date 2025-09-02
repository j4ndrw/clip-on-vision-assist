from src.control_center.models.base import BaseSchema


class IsCameraConnectedResponse(BaseSchema):
    is_camera_connected: bool
