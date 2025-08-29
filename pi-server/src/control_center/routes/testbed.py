from flask import Blueprint, Response
from flask_pydantic import validate

from src.control_center.models.testbed.endpoint.is_camera_connected import IsCameraConnectedResponse
from src.control_center.services.testbed.is_camera_connected import is_camera_connected
from src.control_center.services.testbed.stream_camera_feed import stream_camera_feed

bp = Blueprint("testbed", __name__)


@bp.get("/camera/check")
@validate()
def is_camera_connected_route():
    return IsCameraConnectedResponse(
        is_camera_connected=is_camera_connected()
    ).as_json()

@bp.get("/camera/feed")
@validate()
def stream_camera_feed_route():
    return Response(stream_camera_feed(), mimetype="multipart/x-mixed-replace; boundary=frame")
