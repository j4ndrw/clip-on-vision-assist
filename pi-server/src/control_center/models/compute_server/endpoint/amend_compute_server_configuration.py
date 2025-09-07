from src.control_center.models.base import BaseSchema


class AmendComputeServerConfigurationRequest(BaseSchema):
    endpoint: str
    api_key: str
