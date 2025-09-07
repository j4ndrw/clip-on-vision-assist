import v4l2py

from src.utils.video import (
    get_usb_video_device_ids,
    keep_video_devices_with_mjpg_support,
)


def stream_camera_feed():
    while True:
        device_ids = list(
            filter(keep_video_devices_with_mjpg_support, get_usb_video_device_ids())
        )
        if len(device_ids) == 0:
            return (
                b"HTTP/1.1 500 Internal Server Error\r\n"
                b"Content-Type: text/plain\r\n\r\n"
                b"Error: Could not stream camera feed.\r\n"
            )

        file_id = int(device_ids[0].replace("video", ""))
        try:
            with v4l2py.Device.from_id(file_id) as camera:
                capture = v4l2py.VideoCapture(camera)
                capture.set_format(640, 480, "MJPG")
                for frame in camera:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame.data + b"\r\n\r\n"
                    )
            break
        except Exception as e:
            print(f"Could not stream camera feed - {str(e)}")
            return (
                b"HTTP/1.1 500 Internal Server Error\r\n"
                b"Content-Type: text/plain\r\n\r\n"
                b"Error: Could not stream camera feed.\r\n"
            )
