from src.control_center.models.base import BaseSchema
from src.env import Environment, environment


class AudioCaptureConfig(BaseSchema):
    seconds_per_chunk: int
    max_chunks: int


class SilenceDetectionConfig(BaseSchema):
    min_silence_len_ms: int
    silence_threshold_dbfs: int


class MicrophoneConfig(BaseSchema):
    audio_capture_config: AudioCaptureConfig
    silence_detection_config: SilenceDetectionConfig

    @staticmethod
    def from_environment(environment: Environment = environment):
        env = environment.get()

        peripheral_microphone_capture_seconds = env.get(
            "PERIPHERAL_MICROPHONE_CAPTURE_SECONDS", None
        )
        peripheral_microphone_capture_max_chunks = env.get(
            "PERIPHERAL_MICROPHONE_CAPTURE_MAX_CHUNKS", None
        )
        peripheral_microphone_silence_detection_min_silence_len_ms = env.get(
            "PERIPHERAL_MICROPHONE_SILENCE_DETECTION_MIN_SILENCE_LEN_MS", None
        )
        peripheral_microphone_silence_threshold_dbfs = env.get(
            "PERIPHERAL_MICROPHONE_SILENCE_THRESHOLD_DBFS", None
        )

        assert peripheral_microphone_capture_seconds is not None, (
            "`PERIPHERAL_MICROPHONE_CAPTURE_SECONDS` environment variable is not defined!"
        )
        assert peripheral_microphone_capture_max_chunks is not None, (
            "`PERIPHERAL_MICROPHONE_CAPTURE_MAX_CHUNKS` environment variable is not defined!"
        )
        assert peripheral_microphone_silence_detection_min_silence_len_ms is not None, (
            "`PERIPHERAL_MICROPHONE_SILENCE_DETECTION_MIN_SILENCE_LEN_MS` environment variable is not defined!"
        )
        assert peripheral_microphone_silence_threshold_dbfs is not None, (
            "`PERIPHERAL_MICROPHONE_SILENCE_THRESHOLD_DBFS` environment variable is not defined!"
        )

        audio_capture_config = AudioCaptureConfig(
            seconds_per_chunk=int(peripheral_microphone_capture_seconds),
            max_chunks=int(peripheral_microphone_capture_max_chunks),
        )
        silence_detection_config = SilenceDetectionConfig(
            min_silence_len_ms=int(
                peripheral_microphone_silence_detection_min_silence_len_ms
            ),
            silence_threshold_dbfs=int(peripheral_microphone_silence_threshold_dbfs),
        )

        return MicrophoneConfig(
            audio_capture_config=audio_capture_config,
            silence_detection_config=silence_detection_config,
        )

    def save_to_environment(self, environment: Environment = environment):
        environment.update(
            key="PERIPHERAL_MICROPHONE_CAPTURE_SECONDS",
            value=str(self.audio_capture_config.seconds_per_chunk),
        ).update(
            key="PERIPHERAL_MICROPHONE_CAPTURE_MAX_CHUNKS",
            value=str(self.audio_capture_config.max_chunks),
        ).update(
            key="PERIPHERAL_MICROPHONE_SILENCE_DETECTION_MIN_SILENCE_LEN_MS",
            value=str(self.silence_detection_config.min_silence_len_ms),
        ).update(
            key="PERIPHERAL_MICROPHONE_SILENCE_THRESHOLD_DBFS",
            value=str(self.silence_detection_config.silence_threshold_dbfs),
        )
