from src.control_center.models.base import BaseSchema
from src.control_center.models.compute_server.compute_server import (
    ComputeServerConfigDTO,
)


class GetCurrentComputeServerConfigurationResponse(BaseSchema):
    compute_server_config: ComputeServerConfigDTO
