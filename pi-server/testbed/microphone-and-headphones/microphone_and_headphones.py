import io

import pyaudio
import pydub
import pydub.playback

CHUNK_SIZE = 1024
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2

if __name__ == "__main__":
    stream = pyaudio.PyAudio().open(
        format=SAMPLE_FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        frames_per_buffer=CHUNK_SIZE,
        input=True,
    )
    while True:
        chunk = b"".join(
            stream.read(CHUNK_SIZE)
            for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE))
        )
        segment = pydub.AudioSegment.from_raw(
            io.BytesIO(chunk),
            sample_width=SAMPLE_WIDTH,
            frame_rate=SAMPLE_RATE,
            channels=CHANNELS,
        )
        pydub.playback.play(segment)
