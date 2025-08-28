from src.control_center.models.base import BaseSchema


class AmendLLMConfigurationRequest(BaseSchema):
    model: str
    endpoint: str
    api_key: str
