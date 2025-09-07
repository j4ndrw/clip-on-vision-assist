from typing import Optional

import httpx

from src.control_center.models.error import Error
from src.env import environment


def get_available_llms(*, endpoint: str) -> tuple[list[str], Optional[Error]]:
    try:
        llm_backend_api_key = environment.get().get("LLM_BACKEND_API_KEY", None)
        assert llm_backend_api_key is not None, (
            "`LLM_BACKEND_API_KEY` environment variable not set!"
        )

        compute_server_api_key = environment.get().get("COMPUTE_SERVER_API_KEY", None)
        assert compute_server_api_key is not None, (
            "`COMPUTE_SERVER_API_KEY` environment variable not set!"
        )

        url = environment.get().get("COMPUTE_SERVER_ENDPOINT", None)
        assert url is not None, (
            "`COMPUTE_SERVER_ENDPOINT` environment variable not set!"
        )
        response = httpx.get(
            f"{url}/api/llm/list",
            headers={
                "Content-Type": "application/json",
                "x-llm-backend-endpoint": endpoint,
                "x-llm-backend-api-key": llm_backend_api_key,
                "Authorization": f"Bearer {compute_server_api_key}",
            },
            timeout=httpx.Timeout(10),
        )

        if response.status_code == 400:
            return [], Error(message=response.json()["error"])

        if response.status_code != 200:
            response.raise_for_status()

        return response.json(), None
    except httpx.TimeoutException:
        return [], Error(message="Request to compute server timed out - try again!")
    except httpx.ConnectError:
        return [], Error(message="Could not connect to compute server - try again!")
