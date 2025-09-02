import base64
import json
from typing import Callable

import piper


def as_line(s: str) -> str:
    return f"{s}\n"


def audio_chunk_as_line(_type: str) -> Callable[[piper.AudioChunk], str]:
    return lambda chunk: as_line(
        json.dumps(
            {
                "type": _type,
                "sample_width": chunk.sample_width,
                "frame_rate": chunk.sample_rate,
                "channels": chunk.sample_channels,
                "data": base64.b64encode(chunk.audio_int16_bytes).decode("utf-8"),
            }
        )
    )
