from typing import Optional
from src.control_center.models.base import BaseSchema

class LLMConfig(BaseSchema):
    model: str
    endpoint: Optional[str]
    api_key: str
