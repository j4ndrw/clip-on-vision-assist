import asyncio
from http import HTTPStatus
from flask import Flask, Response, request
from flask_cors import CORS
from flask_pydantic import validate

from src.control_center.models.bluetooth.endpoint.get_bluetooth_devices import GetBluetoothDevicesResponse
from src.control_center.models.bluetooth.endpoint.connect_bluetooth_headphones import ConnectBluetoothHeadphonesRequest
from src.control_center.models.wifi.endpoint.scan_networks import ScanNetworksResponse
from src.control_center.models.wifi.endpoint.connect_to_network import ConnectToNetworkRequest
from src.control_center.services.bluetooth.get_bluetooth_devices import get_bluetooth_devices
from src.control_center.services.bluetooth.connect_bluetooth_headphones import connect_bluetooth_headphones
from src.control_center.services.wifi.scan_networks import scan_networks
from src.control_center.services.wifi.connect_to_network import connect_to_network

loop = asyncio.get_event_loop()
app = Flask(__name__)
CORS(app)


@app.get("/api/bluetooth")
@validate(response_by_alias=True)
def get_bluetooth_devices_route():
    bluetooth_devices = loop.run_until_complete(get_bluetooth_devices())
    return GetBluetoothDevicesResponse(bluetooth_devices=bluetooth_devices).as_json()

@app.post("/api/bluetooth/headphones")
@validate(body=ConnectBluetoothHeadphonesRequest, response_by_alias=True)
def connect_bluetooth_headphones_route():
    body = ConnectBluetoothHeadphonesRequest(**request.json or {})
    err = loop.run_until_complete(connect_bluetooth_headphones(mac_address=body.mac_address))
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(status=HTTPStatus.OK)

@app.get("/api/wifi")
@validate()
def scan_networks_route():
    wifi_networks, err = scan_networks()
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.CONFLICT)
    return ScanNetworksResponse(wifi_networks=wifi_networks).as_json()

@app.post("/api/wifi")
@validate(body=ConnectToNetworkRequest, response_by_alias=True)
def connect_to_network_route():
    body = ConnectToNetworkRequest(**request.json or {})
    err = connect_to_network(ssid=body.ssid, password=body.password)
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(status=200)

if __name__ == "__main__":
    app.run(port=42068) # disappointing
