from http import HTTPStatus
from flask import Response, request
from flask_pydantic import validate

from src.control_center.models.bluetooth.endpoint.get_bluetooth_devices import GetBluetoothDevicesResponse
from src.control_center.models.bluetooth.endpoint.connect_bluetooth_headphones import ConnectBluetoothHeadphonesRequest
from src.control_center.services.bluetooth.get_bluetooth_devices import get_bluetooth_devices
from src.control_center.services.bluetooth.connect_bluetooth_headphones import connect_bluetooth_headphones
from src.control_center.sync.async_loop import async_loop

from flask import Blueprint, Response, request

bp = Blueprint('bluetooth', __name__)

@bp.get("")
@bp.get("/")
@validate()
def get_bluetooth_devices_route():
    bluetooth_devices = async_loop.run_until_complete(get_bluetooth_devices())
    return GetBluetoothDevicesResponse(bluetooth_devices=bluetooth_devices).as_json()

@bp.post("/headphones")
@validate(body=ConnectBluetoothHeadphonesRequest, response_by_alias=True)
def connect_bluetooth_headphones_route():
    body = ConnectBluetoothHeadphonesRequest(**request.json or {})
    err = async_loop.run_until_complete(connect_bluetooth_headphones(mac_address=body.mac_address))
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(status=HTTPStatus.OK)
