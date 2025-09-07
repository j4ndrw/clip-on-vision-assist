import json
import secrets
from typing import Any, Awaitable, Callable, Coroutine
from fastapi import FastAPI, Request, Response
from src.env import environment


class MiddlewareBuilder:
    def __init__(self, app: FastAPI):
        self.app = app

    def add(
        self,
        middleware: Callable[
            [Request, Callable[[Request], Awaitable[Response]]],
            Coroutine[Any, Any, Response],
        ],
    ):
        self.app.middleware("http")(middleware)
        return self


def protect_with_api_key():
    async def handle(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ):
        api_key = environment.get().get("API_KEY", None)
        assert api_key is not None, (
            "You haven't protected your compute server with an API key!!! Aborting..."
        )

        authorization = request.headers.get("Authorization", "")

        if not secrets.compare_digest(f"Bearer {api_key}", authorization or ""):
            return Response(
                json.dumps({"error": "Unauthorized"}),
                status_code=401,
                media_type="application/json",
            )
        return await call_next(request)

    return handle
