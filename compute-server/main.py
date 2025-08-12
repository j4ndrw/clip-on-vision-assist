import base64
import json
from typing import Annotated

import openai

import openwakeword.model
import openwakeword.utils
import vosk
from fastapi import FastAPI, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.camera_frames.camera_frames import camera_frames
from src.llm.client import LLMClient
from src.llm.history import chat_history
from src.microphone_chunks.microphone_chunks import microphone_chunks
from src.requests.microphone_stream import MicrophoneStreamRequest
from src.requests.post_camera_frames import PostCameraFramesRequest
from src.speech.client import speech_client
from src.state_machines.ai_stream_state_machine import (
    AIStreamStateMachineConfig,
    ai_stream_state_machine,
)
from src.systems.ai_system import AISystem

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openwakeword.utils.download_models()

speech_client.use(path="./models/speech/piper/en_US-amy-low.onnx")
wakeword_model = openwakeword.model.Model()
stt_model = vosk.Model(lang="en-us")

@app.post("/api/healthcheck")
async def healthcheck():
    return Response(status_code=200)

@app.post("/api/camera-frames")
async def post_camera_frames(request: PostCameraFramesRequest):
    camera_frames.clear()
    camera_frames.add_many(request.frames)
    return Response(status_code=200)


@app.post("/api/microphone-stream")
async def microphone_stream(request: MicrophoneStreamRequest):
    microphone_chunks.add(base64.b64decode(request.chunk))
    return Response(status_code=200)


@app.post("/api/ai-stream")
async def ai_stream(
    x_llm_backend_endpoint: Annotated[str | None, Header()] = None,
    x_llm_backend_api_key: Annotated[str | None, Header()] = None
):
    if x_llm_backend_endpoint is None:
        return Response(json.dumps({"error": "No LLM backend endpoint provided!"}), status_code=400, media_type="application/json")

    if x_llm_backend_api_key is None:
        return Response(json.dumps({"error": "No LLM backend API key provided!"}), status_code=400, media_type="application/json")

    return StreamingResponse(
        ai_stream_state_machine(
            config=AIStreamStateMachineConfig(
                ai_system=AISystem(
                    speech_client=speech_client,
                    llm_client=LLMClient().use(url=x_llm_backend_endpoint, api_key=x_llm_backend_api_key),
                    wakeword_model=wakeword_model,
                    stt_model=stt_model,
                    chat_history=chat_history,
                    llm="qwen2.5vl:3b",
                )
            )
        ),
        media_type="text/event-stream",
    )

@app.get("/api/llm/list")
async def get_available_llms(
    x_llm_backend_endpoint: Annotated[str | None, Header()] = None,
    x_llm_backend_api_key: Annotated[str | None, Header()] = None
):
    if x_llm_backend_endpoint is None:
        return Response(json.dumps({"error": "No LLM backend endpoint provided!"}), status_code=400, media_type="application/json")

    if x_llm_backend_api_key is None:
        return Response(json.dumps({"error": "No LLM backend API key provided!"}), status_code=400, media_type="application/json")

    try:
        models = LLMClient().use(url=x_llm_backend_endpoint, api_key=x_llm_backend_api_key).get().models.list()
        models = sorted(models, key=lambda m: m.created, reverse=True)
        models = [model.id for model in models]
        return Response(json.dumps(models), status_code=200, media_type="application/json")
    except openai.APIConnectionError:
        return Response(json.dumps({"error": "Could not reach the provided endpoint!"}), status_code=400, media_type="application/json")
