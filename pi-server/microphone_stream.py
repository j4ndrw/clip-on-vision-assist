import signal
import multiprocessing
import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, Callable

import cv2
import httpx
import pyaudio
import pydub
import pydub.playback

CHUNK_SIZE = 1024
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000

def signal_handler(*args, **kwargs):
    print("Main process received termination signal. Exiting...")
    for p in multiprocessing.active_children():
        print(f"Killing worker process {p.pid}")
        p.kill()
    exit(0)


class StreamEventType(Enum):
    LISTEN = "listen"
    TAKE_PICTURES = "take-pictures"
    STOP_LISTENING = "stop-listening"
    AI_SPEECH = "ai-speech"
    UNKNOWN = auto()


@dataclass
class State:
    type: StreamEventType | None = field(default=None)
    task: Callable[[], Any] | None = field(default=None)


class AudioCapturing:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream: pyaudio.Stream = self.p.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            frames_per_buffer=CHUNK_SIZE,
            input=True,
        )

    def read_audio_chunk(self, *, seconds=1):
        return self.as_base64(self.read_audio_chunk_raw(seconds=seconds))

    def read_audio_chunk_raw(self, *, seconds=1):
        if self.stream.is_stopped():
            self.stream.start_stream()

        return b"".join(
                self.stream.read(CHUNK_SIZE)
                for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * seconds))
            )

    def as_base64(self, chunk: bytes) -> str:
        return base64.b64encode(chunk).decode("utf-8")

class VideoCapturing:
    def __init__(self):
        self.camera: cv2.VideoCapture | None = None

    def capture_video(self, *, n=3, fps=1):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise RuntimeError("Camera could not be opened.")

        frames: list[str] = []
        for _ in range(n // fps):
            for _ in range(fps):
                ret, frame = self.camera.read()
                if not ret:
                    raise RuntimeError("Camera capture failed.")

                img_bytes = cv2.imencode(".png", frame)[1].tobytes()
                frames.append(base64.b64encode(img_bytes).decode("utf-8"))

            time.sleep(1 / fps)

        self.camera.release()
        self.camera = None
        return frames


class AIStreamClient:
    def __init__(self):
        self.audio_capturing = AudioCapturing()
        self.video_capturing = VideoCapturing()

    def play_audio_chunk_from_response(self, r: dict[str, Any]) -> None:
        chunk = base64.b64decode(r["data"])
        segment = pydub.AudioSegment.from_raw(
            io.BytesIO(chunk),
            sample_width=r["sample_width"],
            frame_rate=r["frame_rate"],
            channels=r["channels"],
        )
        pydub.playback.play(segment)


class AIStreamTasks:
    def __init__(self, *, client: AIStreamClient):
        self.client = client
        self.audio_capturing_process: multiprocessing.Process | None = None
        self.data_sending_process: multiprocessing.Process | None = None

    def send_microphone_chunks(self):
        def task():
            API.send_audio(self.client.audio_capturing.read_audio_chunk())

        return task

    def take_pictures(self):
        audio_buf = b""

        def capture_audio(audio_buf: bytes):
            audio_buf += self.client.audio_capturing.read_audio_chunk_raw()

        def send_data(audio_buf: bytes):
            API.send_images(self.client.video_capturing.capture_video())
            API.send_audio(self.client.audio_capturing.as_base64(audio_buf))

        def task():
            if self.audio_capturing_process is None:
                self.audio_capturing_process = multiprocessing.Process(
                    target=capture_audio,
                    args=(audio_buf,)
                )
                self.audio_capturing_process.start()

            if self.data_sending_process is None:
                self.data_sending_process = multiprocessing.Process(
                    target=send_data,
                    args=(audio_buf,)
                )
                self.data_sending_process.start()

        return task

    def stop_listening(self):
        def task():
            if not self.client.audio_capturing.stream.is_stopped():
                self.client.audio_capturing.stream.stop_stream()
            if self.client.video_capturing.camera is not None:
                self.client.video_capturing.camera.release()
                self.client.video_capturing.camera = None

            self.audio_capturing_process = None
            self.data_sending_process = None

        return task

    def ai_speech(self, msg: Any):
        def task():
            self.client.play_audio_chunk_from_response(msg)

        return task


class API:
    BASE_URL = "http://localhost:8000/api"

    @classmethod
    def send_audio(cls, chunk: str) -> None:
        r = httpx.post(
            f"{cls.BASE_URL}/microphone-stream",
            content=json.dumps({"chunk": chunk}),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()

    @classmethod
    def send_images(cls, frames_b64: list[str]) -> None:
        r = httpx.post(
            f"{cls.BASE_URL}/camera-frames",
            content=json.dumps({"frames": frames_b64}),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()

    @classmethod
    def ai_stream(cls, *, async_http_client: httpx.AsyncClient):
        return async_http_client.stream(
            "POST", f"{API.BASE_URL}/ai-stream", timeout=httpx.Timeout(None)
        )

async def receive_event(
    ai_stream: AsyncIterator[str], client: AIStreamClient, state: State
) -> State:
    msg = json.loads(await ai_stream.__anext__())
    state.type = StreamEventType(msg["type"])
    tasks = AIStreamTasks(client=client)

    match state.type:
        case StreamEventType.LISTEN:
            state.task = tasks.send_microphone_chunks()
        case StreamEventType.TAKE_PICTURES:
            state.task = tasks.take_pictures()
        case StreamEventType.STOP_LISTENING:
            state.task = tasks.stop_listening()
        case StreamEventType.AI_SPEECH:
            state.task = tasks.ai_speech(msg)
        case _:
            state.task = lambda: None

    return state


async def consume_event(state: State) -> None:
    if not state.type or not state.task:
        return

    state.task()


async def main():
    client = AIStreamClient()
    state = State()

    async with httpx.AsyncClient() as http_client:
        async with API.ai_stream(async_http_client=http_client) as ai_stream:
            iterator = ai_stream.aiter_lines()
            while True:
                print(state.type)
                state, _ = await asyncio.gather(
                    receive_event(iterator, client, state),
                    consume_event(state),
                )


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        asyncio.run(main())
    except KeyboardInterrupt as e:
        signal_handler()
        raise e
