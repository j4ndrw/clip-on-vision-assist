import base64
import io
from typing import Any

import pydub
import pydub.playback

from src.ai_stream_client.io import AIStreamIO


class AIStreamClient:
    def __init__(self):
        self.io = AIStreamIO()

    def play_audio_chunk_from_response(self, r: dict[str, Any]) -> None:
        chunk = base64.b64decode(r["data"])
        segment = pydub.AudioSegment.from_raw(
            io.BytesIO(chunk),
            sample_width=r["sample_width"],
            frame_rate=r["frame_rate"],
            channels=r["channels"],
        )
        pydub.playback.play(segment)
