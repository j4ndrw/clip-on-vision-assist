import v4l2py


def is_camera_connected() -> bool:
    files = list(v4l2py.device.iter_video_files())
    return len([file for file in files if "video" in file.name]) > 0
