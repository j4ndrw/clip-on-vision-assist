from typing import Optional

from src.control_center.models.error import Error
from src.env import environment


def amend_compute_server_configuration(
    *, endpoint: str, api_key: str
) -> Optional[Error]:
    environment.update(key="COMPUTE_SERVER_ENDPOINT", value=endpoint)
    environment.update(key="COMPUTE_SERVER_API_KEY", value=api_key)
