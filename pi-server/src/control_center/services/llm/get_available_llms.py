from typing import Optional

import httpx
from src.constants.api import COMPUTE_SERVER_API_BASE_URL
from src.control_center.models.error import Error
from src.env import environment


def get_available_llms(*, endpoint: str) -> tuple[list[str], Optional[Error]]:
    try:
        llm_backend_api_key = environment.get().get("LLM_BACKEND_API_KEY", None)
        assert llm_backend_api_key is not None, "`LLM_BACKEND_API_KEY` environment variable not set!"

        response = httpx.get(f"{COMPUTE_SERVER_API_BASE_URL}/llm/list", headers={
            "Content-Type": "application/json",
            "x-llm-backend-endpoint": endpoint,
            "x-llm-backend-api-key": llm_backend_api_key
        }, timeout=httpx.Timeout(10))

        if response.status_code == 400:
            return [], Error(message=response.json()["error"])

        if response.status_code != 200:
            response.raise_for_status()

        return response.json(), None
    except httpx.TimeoutException:
        return [], Error(message="Request to compute server timed out - try again!")
    except httpx.ConnectError:
        return [], Error(message="Could not connect to compute server - try again!")
