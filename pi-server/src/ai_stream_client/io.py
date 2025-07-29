import base64
import threading
import time
from typing import Callable

import cv2
import pyaudio

from src.ai_stream_client.constants import (
    CHANNELS,
    CHUNK_SIZE,
    SAMPLE_FORMAT,
    SAMPLE_RATE,
)
from src.utils.generator import StatefulGenerator


class AIStreamIO:
    def __init__(self):
        self.microphone_stream: pyaudio.Stream | None = None

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

    def capture_audio_chunk(self, *, seconds: float = 1):
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

    def capture_audio_until(
        self, until: Callable[[list[bytes]], bool], *, seconds: float = 1, max_chunks=25
    ):
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

    def capture_video(self, *, n=2, fps=1, factor=2):
        def gen():
            camera = cv2.VideoCapture(0)
            camera_frames: list[str] = []

            def worker(camera_frames: list[str]):
                time.sleep(0.1)  # Give the camera a bit of time to power on
                for _ in range(n // fps):
                    for _ in range(fps):
                        ret, frame = camera.read()
                        if not ret:
                            raise RuntimeError("Camera capture failed.")

                        img_bytes = cv2.imencode(".png", frame)[1].tobytes()
                        camera_frames.append(
                            base64.b64encode(img_bytes).decode("utf-8")
                        )

                    time.sleep(factor / fps)

            t = threading.Thread(target=worker, args=(camera_frames,))
            t.daemon = True
            t.start()

            while len(camera_frames) < n:
                yield

            camera.release()
            return camera_frames

        return StatefulGenerator(gen())
