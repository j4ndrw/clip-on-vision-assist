from typing import Optional

import httpx
from src.constants.api import COMPUTE_SERVER_API_BASE_URL
from src.control_center.models.error import Error
from src.env import environment


def get_available_llms() -> tuple[list[str], Optional[Error]]:
    llm_backend_endpoint = environment.get()["LLM_BACKEND_ENDPOINT"]
    llm_backend_api_key = environment.get()["LLM_BACKEND_API_KEY"]

    assert llm_backend_endpoint is not None, "`LLM_BACKEND_ENDPOINT` environment variable not set!"
    assert llm_backend_api_key is not None, "`LLM_BACKEND_API_KEY` environment variable not set!"

    response = httpx.get(f"{COMPUTE_SERVER_API_BASE_URL}/llm/list", headers={
        "Content-Type": "application/json",
        "x-llm-backend-endpoint": llm_backend_endpoint,
        "x-llm-backend-api-key": llm_backend_api_key
    })

    if response.status_code == 400:
        return [], Error(message=response.json()["error"])

    if response.status_code != 200:
        response.raise_for_status()

    return response.json(), None
