from http import HTTPStatus

from flask import Blueprint, Response, json, request
from flask_pydantic import validate

from src.control_center.models.peripheral.endpoint.amend_camera_config import \
    AmendCameraConfigurationRequest
from src.control_center.models.peripheral.endpoint.amend_microphone_config import \
    AmendMicrophoneConfigurationRequest
from src.control_center.models.peripheral.endpoint.get_current_camera_configuration import \
    GetCurrentCameraConfigurationResponse
from src.control_center.models.peripheral.endpoint.get_current_microphone_configuration import \
    GetCurrentMicrophoneConfigurationResponse
from src.control_center.services.peripheral.amend_camera_configuration import \
    amend_camera_configuration
from src.control_center.services.peripheral.amend_microphone_configuration import \
    amend_microphone_configuration
from src.control_center.services.peripheral.get_current_camera_configuration import \
    get_current_camera_configuration
from src.control_center.services.peripheral.get_current_microphone_configuration import \
    get_current_microphone_configuration

bp = Blueprint("peripheral", __name__)


@bp.get("/microphone/config")
@validate()
def get_current_microphone_configuration_route():
    microphone_config = get_current_microphone_configuration()
    return GetCurrentMicrophoneConfigurationResponse(
        microphone_config=microphone_config
    ).as_json()


@bp.get("/camera/config")
@validate()
def get_current_camera_configuration_route():
    camera_config = get_current_camera_configuration()
    return GetCurrentCameraConfigurationResponse(camera_config=camera_config).as_json()


@bp.post("/microphone/config")
@validate(body=AmendMicrophoneConfigurationRequest, response_by_alias=True)
def amend_microphone_configuration_route():
    body = AmendMicrophoneConfigurationRequest(**request.json or {})
    err = amend_microphone_configuration(microphone_config=body.microphone_config)
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(response=json.dumps({}), status=HTTPStatus.OK)


@bp.post("/camera/config")
@validate(body=AmendCameraConfigurationRequest, response_by_alias=True)
def amend_camera_configuration_route():
    body = AmendCameraConfigurationRequest(**request.json or {})
    err = amend_camera_configuration(camera_config=body.camera_config)
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(response=json.dumps({}), status=HTTPStatus.OK)
