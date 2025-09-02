from src.control_center.models.base import BaseSchema
from src.env import Environment, environment


class HotspotConfig(BaseSchema):
    ssid: str
    password: str

    @staticmethod
    def from_environment(environment: Environment = environment):
        env = environment.get()

        hotspot_ssid = env.get("HOTSPOT_SSID", None)
        hotspot_password = env.get("HOTSPOT_PASSWORD", None)

        assert (
            hotspot_ssid is not None
        ), "`PERIPHERAL_MICROPHONE_CAPTURE_SECONDS` environment variable is not defined!"
        assert (
            hotspot_password is not None
        ), "`PERIPHERAL_MICROPHONE_CAPTURE_MAX_CHUNKS` environment variable is not defined!"

        return HotspotConfig(
            ssid=hotspot_ssid,
            password=hotspot_password,
        )

    def save_to_environment(self, environment: Environment = environment):
        environment.update(
            key="HOTSPOT_SSID",
            value=str(self.ssid),
        ).update(
            key="HOTSPOT_PASSWORD",
            value=str(self.password),
        )


class HotspotConfigDTO(BaseSchema):
    ssid: str
