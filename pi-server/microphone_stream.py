import asyncio
import base64
import io
import json
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


class StreamEventType(Enum):
    KEEP_LISTENING = "keep-listening"
    STOP_LISTENING = "stop-listening"
    AI_SPEECH = "ai-speech"
    UNKNOWN = auto()


@dataclass
class State:
    type: StreamEventType | None = field(default=None)
    task: Callable[[], Any] | None = field(default=None)


class AIStreamClient:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.microphone_stream: pyaudio.Stream | None = None

    def start_mic(self) -> None:
        if self.microphone_stream is None:
            self.microphone_stream = self.p.open(
                format=SAMPLE_FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                frames_per_buffer=CHUNK_SIZE,
                input=True,
            )
        if self.microphone_stream.is_stopped():
            self.microphone_stream.start_stream()

    def stop_mic(self) -> None:
        if self.microphone_stream and not self.microphone_stream.is_stopped():
            self.microphone_stream.stop_stream()

    def read_audio_chunk(self) -> bytes:
        assert (
            self.microphone_stream is not None
        ), "Cannot read from the microphone stream - it was not instantiated."

        return b"".join(
            self.microphone_stream.read(CHUNK_SIZE)
            for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE))
        )

    def capture_image(self) -> str:
        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        camera.release()

        if not ret:
            raise RuntimeError("Camera capture failed.")

        img_bytes = cv2.imencode(".png", frame)[1].tobytes()
        return base64.b64encode(img_bytes).decode("utf-8")

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

    def keep_listening(self):
        def task():
            self.client.start_mic()
            API.send_audio(self.client.read_audio_chunk())
            return self.client.microphone_stream

        return task

    def stop_listening_and_send_image(self):
        def task():
            self.client.stop_mic()
            API.send_image(self.client.capture_image())

        return task

    def ai_speech(self, msg: Any):
        def task():
            self.client.play_audio_chunk_from_response(msg)

        return task


class API:
    BASE_URL = "http://localhost:8000/api"

    @classmethod
    def send_audio(cls, chunk: bytes) -> None:
        r = httpx.post(
            f"{cls.BASE_URL}/microphone-stream",
            content=json.dumps({"chunk": base64.b64encode(chunk).decode("utf-8")}),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()

    @classmethod
    def send_image(cls, frame_b64: str) -> None:
        r = httpx.post(
            f"{cls.BASE_URL}/camera-frames",
            content=json.dumps({"frames": [frame_b64]}),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()

    @classmethod
    def ai_stream(cls, *, async_http_client: httpx.AsyncClient):
        return async_http_client.stream(
            "POST", f"{API.BASE_URL}/ai-stream", timeout=httpx.Timeout(None)
        )


async def populate_state(
    ai_stream: AsyncIterator[str], client: AIStreamClient, state: State
) -> State:
    msg = json.loads(await ai_stream.__anext__())
    state.type = StreamEventType(msg["type"])
    tasks = AIStreamTasks(client=client)

    match state.type:
        case StreamEventType.KEEP_LISTENING:
            state.task = tasks.keep_listening()
        case StreamEventType.STOP_LISTENING:
            state.task = tasks.stop_listening_and_send_image()
        case StreamEventType.AI_SPEECH:
            state.task = tasks.ai_speech(msg)
        case _:
            state.task = lambda: None

    return state

async def handle_state(client: AIStreamClient, state: State) -> None:
    if not state.type or not state.task:
        return

    result = state.task()
    match state.type:
        case StreamEventType.KEEP_LISTENING:
            client.microphone_stream = result
        case _:
            pass

    state.type = None
    state.task = None


async def main():
    client = AIStreamClient()
    state = State()

    async with httpx.AsyncClient() as http_client:
        async with API.ai_stream(async_http_client=http_client) as ai_stream:
            iterator = ai_stream.aiter_lines()
            while True:
                state, _ = await asyncio.gather(
                    populate_state(iterator, client, state),
                    handle_state(client, state),
                )


if __name__ == "__main__":
    asyncio.run(main())
