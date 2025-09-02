import subprocess


def get_usb_video_device_ids() -> list[str]:
    command = ["ls", "-la", "/sys/class/video4linux"]
    try:
        output = subprocess.check_output(command, text=True)
        lines = output.splitlines()
        usb_device_ids = [
            line.split()[-1].rsplit("/", maxsplit=1)[-1]
            for line in lines
            if "usb" in line.lower()
        ]
        return usb_device_ids
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return []


def keep_video_devices_with_mjpg_support(device: str) -> bool:
    try:
        command = ["v4l2-ctl", f"--device=/dev/{device}", "-D", "--list-formats"]
        output = subprocess.check_output(command, text=True)
        if "MJPG" in output:
            return True
    except subprocess.CalledProcessError as e:
        print(f"Error accessing device {device}: {e}")
    return False
