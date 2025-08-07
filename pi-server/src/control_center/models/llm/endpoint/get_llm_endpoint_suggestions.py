from src.control_center.models.base import BaseSchema


class GetLLMEndpointSuggestionsResponse(BaseSchema):
    endpoint_suggestions: list[str]
