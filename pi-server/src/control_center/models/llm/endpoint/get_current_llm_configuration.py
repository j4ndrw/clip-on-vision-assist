from src.control_center.models.base import BaseSchema
from src.control_center.models.llm.llm import LLMConfig


class GetCurrentLLMConfigurationResponse(BaseSchema):
    llm_config: LLMConfig
