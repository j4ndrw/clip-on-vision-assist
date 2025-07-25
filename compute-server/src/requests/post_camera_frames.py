from pydantic import BaseModel


class PostCameraFramesRequest(BaseModel):
    frames: list[str]
