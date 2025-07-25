from pydantic import BaseModel


class MicrophoneStreamRequest(BaseModel):
    chunk: str
