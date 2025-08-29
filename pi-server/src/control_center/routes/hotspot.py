from http import HTTPStatus

from flask import Blueprint, Response, json, request
from flask_pydantic import validate

from src.control_center.models.hotspot.endpoint.amend_hotspot_config import (
    AmendHotspotConfigurationRequest,
)
from src.control_center.models.hotspot.endpoint.get_current_hotspot_configuration import (
    GetCurrentHotspotConfigurationResponse,
)
from src.control_center.models.hotspot.hotspot import HotspotConfigDTO
from src.control_center.services.hotspot.amend_hotspot_configuration import (
    amend_hotspot_configuration,
)
from src.control_center.services.hotspot.get_current_hotspot_configuration import (
    get_current_hotspot_configuration,
)

bp = Blueprint("hotspot", __name__)


@bp.get("/config")
@validate()
def get_current_hotspot_configuration_route():
    hotspot_config = get_current_hotspot_configuration()
    return GetCurrentHotspotConfigurationResponse(
        hotspot_config=HotspotConfigDTO(ssid=hotspot_config.ssid)
    ).as_json()


@bp.post("/config")
@validate(body=AmendHotspotConfigurationRequest, response_by_alias=True)
def amend_hotspot_configuration_route():
    body = AmendHotspotConfigurationRequest(**request.json or {})
    err = amend_hotspot_configuration(hotspot_config=body.hotspot_config)
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(response=json.dumps({}), status=HTTPStatus.OK)
