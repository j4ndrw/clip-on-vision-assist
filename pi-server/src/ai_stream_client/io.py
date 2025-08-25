import base64
import threading
import time
from typing import Callable, Optional

import v4l2py
import pyaudio

from src.ai_stream_client.constants import (
    CHANNELS,
    CHUNK_SIZE,
    SAMPLE_FORMAT,
    SAMPLE_RATE,
)
from src.control_center.models.peripheral.camera import CameraConfig
from src.control_center.models.peripheral.microphone import MicrophoneConfig
from src.utils.generator import StatefulGenerator


class AIStreamIO:
    def __init__(
        self,
        *,
        microphone_config: MicrophoneConfig = MicrophoneConfig.from_environment(),
        camera_config: CameraConfig = CameraConfig.from_environment()
    ):
        self.microphone_stream: Optional[pyaudio.Stream] = None
        self.microphone_config = microphone_config
        self.camera_config = camera_config

    def open_audio_stream(self):
        self.microphone_stream = pyaudio.PyAudio().open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=CHUNK_SIZE,
            input=True,
        )

    def close_audio_stream(self):
        if self.microphone_stream is not None:
            self.microphone_stream.stop_stream()
            self.microphone_stream.close()
            self.microphone_stream = None

    def restart_audio_stream(self):
        self.close_audio_stream()
        self.open_audio_stream()

    def capture_audio_chunk(self):
        seconds = self.microphone_config.audio_capture_config.seconds_per_chunk

        def gen():
            ret: list[bytes] = []

            def worker(audio_chunk: list[bytes]):
                assert self.microphone_stream is not None
                audio_chunk.append(
                    b"".join(
                        self.microphone_stream.read(CHUNK_SIZE)
                        for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * seconds))
                    )
                )

            t = threading.Thread(target=worker, args=(ret,))
            t.daemon = True
            t.start()

            while len(ret) == 0:
                yield None
            return ret[0]

        return StatefulGenerator(gen())

    def capture_audio_until(self, until: Callable[[list[bytes]], bool]):
        seconds = self.microphone_config.audio_capture_config.seconds_per_chunk
        max_chunks = self.microphone_config.audio_capture_config.max_chunks

        def gen():
            ret: list[bytes] = []
            audio_chunks: list[bytes] = []

            def worker(audio_chunks: list[bytes]):
                assert self.microphone_stream is not None
                while not until(audio_chunks) or len(audio_chunks) >= max_chunks:
                    audio_chunks.append(
                        b"".join(
                            self.microphone_stream.read(CHUNK_SIZE)
                            for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * seconds))
                        )
                    )
                ret.extend(audio_chunks)

            t = threading.Thread(target=worker, args=(audio_chunks,))
            t.daemon = True
            t.start()

            while len(ret) == 0:
                yield None
            return ret

        return StatefulGenerator(gen())

    def capture_video(self):
        n = self.camera_config.num_frames_to_capture
        fps = self.camera_config.fps
        factor = self.camera_config.wait_for_next_batch_factor

        def gen():
            with v4l2py.Device.from_id(0) as camera:
                capture = v4l2py.VideoCapture(camera)
                capture.set_format(640, 480, "MJPG")

                frames = 0
                frames_in_batch = 0
                batch_size = n // fps
                for frame in camera:
                    yield base64.b64encode(frame.data).decode("utf-8")
                    frames += 1

                    if frames >= n:
                        break

                    if frames_in_batch + 1 >= batch_size:
                        time.sleep(factor / fps)

                    frames_in_batch += 1
                    frames_in_batch %= batch_size

                camera.close()

        return StatefulGenerator(gen())
