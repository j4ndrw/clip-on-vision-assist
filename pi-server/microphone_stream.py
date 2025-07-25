import asyncio
import base64
import io
import json
from typing import Any, AsyncIterator

import cv2
import httpx
import pyaudio
import pydub
import pydub.playback

chunk_size = 1024
sample_format = pyaudio.paInt16
channels = 1
sample_rate = 16000

class ComputeServerHandlers:
    KEEP_LISTENING_TYPE = "keep-listening"
    STOP_LISTENING_TYPE = "stop-listening"
    AI_SPEECH_TYPE = "ai-speech"

    @staticmethod
    def handle_keep_listening(
        p: pyaudio.PyAudio, dependencies: dict[str, Any]
    ):
        microphone_stream: pyaudio.Stream | None = dependencies["microphone_stream"]
        if microphone_stream is None:
            microphone_stream = p.open(
                format=sample_format,
                channels=channels,
                rate=sample_rate,
                frames_per_buffer=chunk_size,
                input=True,
            )
        if microphone_stream.is_stopped():
            microphone_stream.start_stream()

        buf = b""
        for _ in range(0, int(sample_rate / chunk_size)):
            chunk = microphone_stream.read(chunk_size)
            buf += chunk

        ComputeServerRequests.send_microphone_input(buf)
        return microphone_stream

    @staticmethod
    def handle_stop_listening(dependencies: dict[str, Any]):
        microphone_stream = dependencies["microphone_stream"]

        camera = cv2.VideoCapture(0)
        ret, frame = camera.read()
        camera.release()

        if not ret:
            raise Exception(
                "Could not take picture with camera - something went wrong..."
            )

        frame = base64.b64encode(cv2.imencode(".png", frame)[1].tobytes()).decode(
            "utf-8"
        )
        ComputeServerRequests.send_camera_frames([frame])

        if microphone_stream is not None:
            microphone_stream.stop_stream()

    @staticmethod
    def handle_ai_speech(ai_stream_response: dict[str, Any]):
        chunk_data = base64.b64decode(ai_stream_response["data"])
        chunk_sample_width = ai_stream_response["sample_width"]
        chunk_frame_rate = ai_stream_response["sample_rate"]
        chunk_channels = ai_stream_response["sample_channels"]

        segment = pydub.AudioSegment.from_raw(
            io.BytesIO(chunk_data),
            sample_width=chunk_sample_width,
            frame_rate=chunk_frame_rate,
            channels=chunk_channels,
        )
        pydub.playback.play(segment)


class ComputeServerRequests:
    SEND_CAMERA_FRAMES_ENDPOINT = "http://localhost:8000/api/camera-frames"
    SEND_MICROPHONE_INPUT_ENDPOINT = "http://localhost:8000/api/microphone-stream"
    READ_AI_RESPONSE_ENDPOINT = "http://localhost:8000/api/ai-stream"

    @staticmethod
    def send_microphone_input(chunk: bytes):
        response = httpx.post(
            ComputeServerRequests.SEND_MICROPHONE_INPUT_ENDPOINT,
            content=json.dumps({"chunk": base64.b64encode(chunk).decode("utf-8")}),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            raise Exception(
                "Something wrong happened while sending mic data to compute server..."
            )

    @staticmethod
    def send_camera_frames(frames: list[str]):
        response = httpx.post(
            ComputeServerRequests.SEND_CAMERA_FRAMES_ENDPOINT,
            content=json.dumps({"frames": frames}),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            raise Exception(
                "Something wrong happened while sending mic data to compute server..."
            )

async def populate_ai_stream_state_machine(
    ai_stream: AsyncIterator[str],
    p: pyaudio.PyAudio,
    state_machine: dict[str, Any],
    dependencies: dict[str, Any],
):
    ai_stream_response = json.loads(await ai_stream.__anext__())

    state = ai_stream_response["type"]
    task = None

    match ai_stream_response["type"]:
        case ComputeServerHandlers.KEEP_LISTENING_TYPE:
            task = lambda: ComputeServerHandlers.handle_keep_listening(
                p, dependencies
            )

        case ComputeServerHandlers.STOP_LISTENING_TYPE:
            task = lambda: ComputeServerHandlers.handle_stop_listening(dependencies)

        case ComputeServerHandlers.AI_SPEECH_TYPE:
            task = lambda: ComputeServerHandlers.handle_ai_speech(ai_stream_response)

    state_machine["state"] = state
    state_machine["task"] = task

async def handle_ai_stream_state_machine(
    state_machine: dict[str, Any],
    dependencies: dict[str, Any]
):
    state = state_machine["state"]
    task = state_machine["task"]
    if state is None or task is None:
        return

    if state == ComputeServerHandlers.KEEP_LISTENING_TYPE:
        dependencies["microphone_stream"] = task()
    else:
        task()

    state_machine["state"] = None
    state_machine["task"] = None

async def main():
    p = pyaudio.PyAudio()
    dependencies = {"microphone_stream": None}
    state_machine = {"state": None, "task": None}

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            ComputeServerRequests.READ_AI_RESPONSE_ENDPOINT,
            timeout=httpx.Timeout(None),
        ) as ai_stream:
            iterator = ai_stream.aiter_lines()
            while True:
                await asyncio.gather(
                    populate_ai_stream_state_machine(iterator, p, state_machine, dependencies),
                    handle_ai_stream_state_machine(state_machine, dependencies),
                )

if __name__ == "__main__":
    asyncio.run(main())
