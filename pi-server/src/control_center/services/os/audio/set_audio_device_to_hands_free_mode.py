import subprocess

def set_audio_device_to_hands_free_mode():
    get_audio_card_process = subprocess.Popen(
        "pactl list cards | grep 'Name:' | awk -F': ' '{print $2}'",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = get_audio_card_process.communicate()

    audio_card = stdout.decode().strip()
    if not audio_card:
        raise Exception("No audio card found.")

    set_audio_device_profile_process = subprocess.Popen(
        f"pactl set-card-profile {audio_card} headset_head_unit",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    _, stderr = set_audio_device_profile_process.communicate()

    if stderr:
        raise Exception(stderr.decode().strip())
