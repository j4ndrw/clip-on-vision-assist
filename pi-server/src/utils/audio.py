import io
import os
from typing import Callable

import pydub
import pydub.playback
import pydub.silence

from src.ai_stream_client.constants import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH
from src.control_center.models.peripheral.microphone import SilenceDetectionConfig


def silence_detected(
    *, config: SilenceDetectionConfig
) -> Callable[[list[bytes]], bool]:
    min_silence_len_ms = config.min_silence_len_ms
    silence_threshold_dBFS = config.silence_threshold_dbfs

    def ret(audio_chunks: list[bytes]) -> bool:
        if len(audio_chunks) == 0:
            return False

        buf = b"".join(audio_chunks)

        segment = pydub.AudioSegment.from_raw(
            io.BytesIO(buf),
            sample_width=SAMPLE_WIDTH,
            frame_rate=SAMPLE_RATE,
            channels=CHANNELS,
        )

        silence_ranges = pydub.silence.detect_silence(
            segment,
            min_silence_len=min_silence_len_ms,
            silence_thresh=silence_threshold_dBFS,
        )
        segment = (
            segment[silence_ranges[0][0] :]
            if silence_ranges and silence_ranges[0][0] > 0
            else segment
        )
        silence_ranges = pydub.silence.detect_silence(
            segment,
            min_silence_len=min_silence_len_ms,
            silence_thresh=silence_threshold_dBFS,
        )
        return len(silence_ranges) > 0

    return ret


def play_asset(name: str, *, volume_db: int = -3):
    segment = (
        pydub.AudioSegment.from_mp3(
            os.path.join(os.curdir, "assets", "sounds", f"{name}.mp3")
        )
        + volume_db
    )
    pydub.playback.play(segment)
