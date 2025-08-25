from typing import Optional

from src.control_center.models.peripheral.microphone import MicrophoneConfig
from src.control_center.models.error import Error


def amend_microphone_configuration(
    *,
    microphone_config: MicrophoneConfig
) -> Optional[Error]:
    microphone_config.save_to_environment()
