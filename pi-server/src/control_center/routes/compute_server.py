from http import HTTPStatus

from flask import Blueprint, Response, json, request
from flask_pydantic import validate

from src.control_center.models.compute_server.compute_server import (
    ComputeServerConfigDTO,
)
from src.control_center.models.compute_server.endpoint.amend_compute_server_configuration import (
    AmendComputeServerConfigurationRequest,
)
from src.control_center.models.compute_server.endpoint.get_current_compute_server_configuration import (
    GetCurrentComputeServerConfigurationResponse,
)
from src.control_center.services.compute_server.amend_compute_server_configuration import (
    amend_compute_server_configuration,
)
from src.control_center.services.compute_server.get_current_compute_server_configuration import (
    get_current_compute_server_configuration,
)

bp = Blueprint("compute_server", __name__)


@bp.get("/config")
@validate()
def get_current_compute_server_configuration_route():
    compute_server_config = get_current_compute_server_configuration()
    return GetCurrentComputeServerConfigurationResponse(
        compute_server_config=ComputeServerConfigDTO(
            endpoint=compute_server_config.endpoint
        )
    ).as_json()


@bp.post("/config")
@validate(body=AmendComputeServerConfigurationRequest, response_by_alias=True)
def amend_compute_server_configuration_route():
    body = AmendComputeServerConfigurationRequest(**request.json or {})
    err = amend_compute_server_configuration(
        endpoint=body.endpoint, api_key=body.api_key
    )
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(response=json.dumps({}), status=HTTPStatus.OK)
