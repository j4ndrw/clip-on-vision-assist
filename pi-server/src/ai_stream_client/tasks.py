import time
from typing import Any

from src.ai_stream_client.api import API
from src.ai_stream_client.assets import (
    PROMPT_CAPTURED,
    READY_TO_LISTEN,
    WAKEWORD_DETECTED,
)
from src.ai_stream_client.client import AIStreamClient
from src.utils.audio import play_asset, silence_detected
from src.utils.data import as_base64


class AIStreamTasks:
    def __init__(self, *, client: AIStreamClient):
        self.client = client

    def capture_wakeword(self):
        def task():
            if self.client.io.microphone_stream is None:
                self.client.io.open_audio_stream()
                play_asset(READY_TO_LISTEN, volume_db=-12)
                time.sleep(0.2)

            audio_chunk = b"".join(
                self.client.io.capture_audio_until(lambda chunks: len(chunks) > 1)
                .consume()
                .ret
            )
            API.send_microphone_audio(as_base64(audio_chunk))

        return task

    def capture_prompt(self):
        def task():
            play_asset(WAKEWORD_DETECTED)
            self.client.io.restart_audio_stream()

            audio_buf = b""

            video_stream = self.client.io.capture_video()
            for _ in video_stream:
                audio_chunk = self.client.io.capture_audio_chunk().consume().ret
                audio_buf += audio_chunk

            API.send_camera_frames(video_stream.ret)

            audio_buf += b"".join(
                self.client.io.capture_audio_until(silence_detected()).consume().ret
            )
            API.send_microphone_audio(as_base64(audio_buf))

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
