from src.control_center.models.peripheral.microphone import MicrophoneConfig


def get_current_microphone_configuration() -> MicrophoneConfig:
    return MicrophoneConfig.from_environment()
