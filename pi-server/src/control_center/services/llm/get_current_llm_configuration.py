from typing import Optional

from src.control_center.models.llm.llm import LLMConfig
from src.env import environment


def get_current_llm_configuration() -> LLMConfig:
    model = environment.get()["LLM"]
    endpoint: Optional[str] = environment.get()["LLM_BACKEND_ENDPOINT"]
    api_key: Optional[str] = environment.get()["LLM_BACKEND_API_KEY"]

    assert model is not None, "`LLM` environment variable not set!"
    assert endpoint is not None, "`LLM_BACKEND_ENDPOINT` environment variable not set!"

    return LLMConfig(model=model, endpoint=endpoint, api_key=api_key or "")
