import base64
import time
from typing import Callable, Mapping, Optional

import pyaudio
import v4l2py

from src.ai_stream_client.constants import (CHANNELS, CHUNK_SIZE,
                                            SAMPLE_FORMAT, SAMPLE_RATE)
from src.control_center.models.peripheral.camera import CameraConfig
from src.control_center.models.peripheral.microphone import MicrophoneConfig
from src.utils.generator import StatefulGenerator
from src.utils.video import get_usb_video_device_ids, keep_video_devices_with_mjpg_support


class AIStreamIO:
    def __init__(
        self,
        *,
        microphone_config: MicrophoneConfig = MicrophoneConfig.from_environment(),
        camera_config: CameraConfig = CameraConfig.from_environment(),
        on_microphone_chunk: Optional[Callable[[bytes], None]] = None
    ):
        self.microphone_buffer: list[bytes] = []
        self.microphone_stream: Optional[pyaudio.Stream] = None
        self.on_microphone_chunk = on_microphone_chunk
        self.microphone_stream_start_time: Optional[float] = None
        self.microphone_config = microphone_config

        self.camera_config = camera_config

    def open_audio_stream(self):
        self.microphone_stream = pyaudio.PyAudio().open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=CHUNK_SIZE,
            input=True,
            stream_callback=self.capture_audio_chunk,
        )

    def close_audio_stream(self):
        if self.microphone_stream is not None:
            self.microphone_stream.stop_stream()
            self.microphone_stream.close()
            self.microphone_stream = None

        self.microphone_stream_start_time = None
        self.on_microphone_chunk = None
        self.microphone_buffer = []

    def restart_audio_stream(self):
        self.close_audio_stream()
        self.open_audio_stream()

    def capture_audio_chunk(
        self,
        in_data: Optional[bytes],
        frame_count: int,
        time_info: Mapping[str, float],
        status: int,
    ) -> tuple[Optional[bytes], int]:
        if not self.on_microphone_chunk:
            self.microphone_stream_start_time = None
            return (in_data, pyaudio.paContinue)

        if self.microphone_stream_start_time is None:
            self.microphone_stream_start_time = self.microphone_stream_start_time or time.time()

        if not in_data:
            return (in_data, pyaudio.paContinue)

        elapsed_time = time.time() - self.microphone_stream_start_time
        seconds = self.microphone_config.audio_capture_config.seconds_per_chunk

        if elapsed_time < seconds:
            self.microphone_buffer.append(in_data)
            return (in_data, pyaudio.paContinue)

        audio_chunk = b"".join(self.microphone_buffer)
        self.microphone_buffer.clear()
        self.microphone_stream_start_time = None
        self.on_microphone_chunk(audio_chunk)

        return (in_data, pyaudio.paContinue)

    def capture_video(self):
        n = self.camera_config.num_frames_to_capture
        fps = self.camera_config.fps
        factor = self.camera_config.wait_for_next_batch_factor

        def gen():
            while True:
                device_ids = list(
                    filter(
                        keep_video_devices_with_mjpg_support, get_usb_video_device_ids()
                    )
                )
                if len(device_ids) == 0:
                    raise Exception("No video device found - cannot capture video")

                file_id = int(device_ids[0].replace("video", ""))
                try:
                    with v4l2py.Device.from_id(file_id) as camera:
                        capture = v4l2py.VideoCapture(camera)
                        capture.set_format(640, 480, "MJPG")

                        time.sleep(0.2) # Give camera time to warm up

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
                    break
                except Exception as e:
                    print(f"Failed to capture video. Reason: {str(e)} - Trying another device")

        return StatefulGenerator(gen())
