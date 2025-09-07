from src.control_center.models.base import BaseSchema


class ComputeServerConfig(BaseSchema):
    endpoint: str
    api_key: str


class ComputeServerConfigDTO(BaseSchema):
    endpoint: str
