from src.control_center.models.base import BaseSchema
from src.control_center.models.peripheral.microphone import MicrophoneConfig

class AmendMicrophoneConfigurationRequest(BaseSchema):
    microphone_config: MicrophoneConfig
