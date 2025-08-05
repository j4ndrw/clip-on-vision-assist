import asyncio
from http import HTTPStatus
from flask import Flask, Response
from flask_cors import CORS
from flask_pydantic import validate

from src.control_center.models.bluetooth.endpoint.get_bluetooth_devices import GetBluetoothDevicesResponse
from src.control_center.models.bluetooth.endpoint.connect_bluetooth_headphones import ConnectBluetoothHeadphonesRequest
from src.control_center.services.bluetooth.get_bluetooth_devices import get_bluetooth_devices
from src.control_center.services.bluetooth.connect_bluetooth_headphones import connect_bluetooth_headphones

loop = asyncio.get_event_loop()
app = Flask(__name__)
CORS(app)


@app.get("/api/bluetooth")
@validate()
def get_bluetooth_devices_route():
    bluetooth_devices = loop.run_until_complete(get_bluetooth_devices())
    return GetBluetoothDevicesResponse(bluetooth_devices=bluetooth_devices).as_json()

@app.post("/api/bluetooth/headphones")
@validate()
def connect_bluetooth_headphones_route(request: ConnectBluetoothHeadphonesRequest):
    err = loop.run_until_complete(connect_bluetooth_headphones(mac_address=request.mac_address))
    if err is not None:
        return Response(response=err.as_json(), status=HTTPStatus.BAD_REQUEST)
    return Response(status=HTTPStatus.OK)

if __name__ == "__main__":
    app.run(port=42068) # disappointing
