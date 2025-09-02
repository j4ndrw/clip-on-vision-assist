import time
from typing import Any

from src.ai_stream_client.api import API
from src.ai_stream_client.assets import (PROMPT_CAPTURED, READY_TO_LISTEN,
                                         WAKEWORD_DETECTED)
from src.ai_stream_client.client import AIStreamClient
from src.utils.audio import play_asset, silence_detected
from src.utils.data import as_base64


class WakewordBasedStateMachineTasks:
    def __init__(self, *, client: AIStreamClient):
        self.client = client

    def capture_wakeword(self):
        def task():
            if self.client.io.microphone_stream is None:
                self.client.io.open_audio_stream()
                play_asset(READY_TO_LISTEN, volume_db=-12)
                time.sleep(0.2)

            audio_buf: list[bytes] = []
            self.client.io.on_microphone_chunk = audio_buf.append
            while len(audio_buf) <= 1:
                pass
            self.client.io.on_microphone_chunk = None

            API.send_microphone_audio(as_base64(b"".join(audio_buf)))

        return task

    def capture_prompt(self):
        def task():
            play_asset(WAKEWORD_DETECTED)
            self.client.io.restart_audio_stream()

            audio_buf: list[bytes] = []
            video_buf: list[str] = []

            self.client.io.on_microphone_chunk = audio_buf.append

            video_stream = self.client.io.capture_video()
            for frame in video_stream:
                video_buf.append(frame)

            is_audio_captured = lambda: 0 < len(audio_buf) < self.client.io.microphone_config.audio_capture_config.max_chunks
            while (
                not silence_detected(
                    config=self.client.io.microphone_config.silence_detection_config
                )
                and is_audio_captured()
            ):
                pass

            self.client.io.on_microphone_chunk = None

            API.send_camera_frames(video_buf)
            API.send_microphone_audio(as_base64(b"".join(audio_buf)))
        return task

    def stall(self):
        def task():
            play_asset(PROMPT_CAPTURED, volume_db=-4)
            self.client.io.close_audio_stream()

        return task

    def ai_speech(self, msg: Any):
        def task():
            self.client.play_audio_chunk_from_response(msg)

        return task

    def done(self):
        def task():
            return

        return task
