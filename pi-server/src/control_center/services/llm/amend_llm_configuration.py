from typing import Optional

from src.control_center.models.error import Error
from src.env import environment


def amend_llm_configuration(
    *, endpoint: str, model: str, api_key: str
) -> Optional[Error]:
    environment.update(key="LLM", value=model)
    environment.update(key="LLM_BACKEND_ENDPOINT", value=endpoint)
    environment.update(key="LLM_BACKEND_API_KEY", value=api_key)
