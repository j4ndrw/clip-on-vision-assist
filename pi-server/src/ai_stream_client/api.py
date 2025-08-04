import json

import httpx


class API:
    BASE_URL = "http://192.168.100.16:8000/api"
    # BASE_URL = "http://localhost:8000/api"

    @classmethod
    def healthcheck(cls) -> None:
        r = httpx.post(f"{cls.BASE_URL}/healthcheck", timeout=httpx.Timeout(3.0))
        if r.status_code != 200:
            raise httpx.ConnectError("API failed healthcheck")

    @classmethod
    def send_microphone_audio(cls, chunk: str) -> None:
        r = httpx.post(
            f"{cls.BASE_URL}/microphone-stream",
            content=json.dumps({"chunk": chunk}),
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(None)
        )
        r.raise_for_status()

    @classmethod
    def send_camera_frames(cls, frames_b64: list[str]) -> None:
        r = httpx.post(
            f"{cls.BASE_URL}/camera-frames",
            content=json.dumps({"frames": frames_b64}),
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(None)
        )
        r.raise_for_status()

    @classmethod
    def ai_stream(cls, *, async_http_client: httpx.AsyncClient):
        return async_http_client.stream(
            "POST", f"{API.BASE_URL}/ai-stream", timeout=httpx.Timeout(None)
        )
