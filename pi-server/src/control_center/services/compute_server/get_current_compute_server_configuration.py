from typing import Optional

from src.control_center.models.compute_server.compute_server import ComputeServerConfig
from src.env import environment


def get_current_compute_server_configuration() -> ComputeServerConfig:
    endpoint: Optional[str] = environment.get()["COMPUTE_SERVER_ENDPOINT"]
    api_key: Optional[str] = environment.get()["COMPUTE_SERVER_API_KEY"]

    assert endpoint is not None, (
        "`COMPUTE_SERVER_ENDPOINT` environment variable not set!"
    )
    assert api_key is not None, "`COMPUTE_SERVER_API_KEY` environment variable not set!"

    return ComputeServerConfig(endpoint=endpoint, api_key=api_key)
