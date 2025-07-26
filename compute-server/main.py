import base64

import openwakeword.model
import openwakeword.utils
import vosk
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.camera_frames.camera_frames import camera_frames
from src.generators.ai_stream_generator import ai_stream_generator
from src.llm.client import llm_client
from src.llm.endpoints import OLLAMA_ENDPOINT
from src.llm.history import chat_history
from src.microphone_chunks.microphone_chunks import microphone_chunks
from src.requests.microphone_stream import MicrophoneStreamRequest
from src.requests.post_camera_frames import PostCameraFramesRequest
from src.speech.client import speech_client

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openwakeword.utils.download_models()

llm_client.use(url=OLLAMA_ENDPOINT)
speech_client.use(path="./models/speech/piper/en_US-amy-low.onnx")
wakeword_model = openwakeword.model.Model()
stt_model = vosk.Model(lang="en-us")


@app.post("/api/camera-frames")
async def post_camera_frames(request: PostCameraFramesRequest):
    camera_frames.clear()
    camera_frames.extend(request.frames)
    return Response(status_code=200)


@app.post("/api/microphone-stream")
async def microphone_stream(request: MicrophoneStreamRequest):
    microphone_chunks.add(base64.b64decode(request.chunk))
    return Response(status_code=200)


@app.post("/api/ai-stream")
async def ai_stream():
    return StreamingResponse(
        ai_stream_generator(
            speech_client=speech_client,
            llm_client=llm_client,
            wakeword_model=wakeword_model,
            stt_model=stt_model,
            chat_history=chat_history,
            llm="qwen2.5vl:3b",
        ),
        media_type="text/event-stream",
    )
