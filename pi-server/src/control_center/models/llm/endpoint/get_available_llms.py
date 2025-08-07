from src.control_center.models.base import BaseSchema


class GetAvailableLLMsResponse(BaseSchema):
    llms: list[str]
