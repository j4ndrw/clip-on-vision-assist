import json

import httpx

from src.env import environment


class API:
    @classmethod
    def http_client(cls):
        compute_server_api_key = environment.get().get("COMPUTE_SERVER_API_KEY", None)
        assert compute_server_api_key is not None, (
            "`COMPUTE_SERVER_API_KEY` environment variable not set!"
        )

        url = environment.get().get("COMPUTE_SERVER_ENDPOINT", None)
        assert url is not None, (
            "`COMPUTE_SERVER_ENDPOINT` environment variable not set!"
        )

        return httpx.Client(
            base_url=httpx.URL(f"{url}/api"),
            headers={"Authorization": f"Bearer {compute_server_api_key}"},
        )

    @classmethod
    def healthcheck(cls) -> None:
        r = cls.http_client().post("/healthcheck", timeout=httpx.Timeout(3.0))
        if r.status_code != 200:
            raise httpx.ConnectError("API failed healthcheck")

    @classmethod
    def send_microphone_audio(cls, chunk: str) -> None:
        r = cls.http_client().post(
            "/microphone-stream",
            content=json.dumps({"chunk": chunk}),
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(None),
        )
        r.raise_for_status()

    @classmethod
    def send_camera_frames(cls, frames_b64: list[str]) -> None:
        r = cls.http_client().post(
            "/camera-frames",
            content=json.dumps({"frames": frames_b64}),
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(None),
        )
        r.raise_for_status()

    @classmethod
    def ai_stream(cls, *, async_http_client: httpx.AsyncClient):
        url = environment.get().get("COMPUTE_SERVER_ENDPOINT", None)
        assert url is not None, (
            "`COMPUTE_SERVER_ENDPOINT` environment variable not set!"
        )

        return async_http_client.stream(
            "POST", f"{url}/api/ai-stream", timeout=httpx.Timeout(None)
        )
