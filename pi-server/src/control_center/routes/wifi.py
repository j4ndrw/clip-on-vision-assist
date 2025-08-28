from http import HTTPStatus

from flask import Blueprint, Response, json, request
from flask_pydantic import validate

from src.control_center.models.wifi.endpoint.connect_to_network import \
    ConnectToNetworkRequest
from src.control_center.models.wifi.endpoint.scan_networks import \
    ScanNetworksResponse
from src.control_center.services.wifi.connect_to_network import \
    connect_to_network
from src.control_center.services.wifi.scan_networks import scan_networks

bp = Blueprint("wifi", __name__)


@bp.get("")
@bp.get("/")
@validate()
def scan_networks_route():
    wifi_networks, err = scan_networks()
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.CONFLICT)
    return ScanNetworksResponse(wifi_networks=wifi_networks).as_json()


@bp.post("")
@bp.post("/")
@validate(body=ConnectToNetworkRequest, response_by_alias=True)
def connect_to_network_route():
    body = ConnectToNetworkRequest(**request.json or {})
    err = connect_to_network(ssid=body.ssid, password=body.password)
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(response=json.dumps({}), status=HTTPStatus.OK)
