import base64
import io
import json

import numpy as np
import openwakeword.model
import pydub
import pydub.playback
import pydub.silence
import vosk

from src.camera_frames.camera_frames import camera_frames
from src.llm.client import LLMClient
from src.llm.history import ChatHistory
from src.llm.prompts import SYSTEM_PROMPT
from src.microphone_chunks.microphone_chunks import microphone_chunks
from src.speech.client import SpeechClient
from src.state.state import states
from src.utils.stream import as_line

KEEP_LISTENING = as_line(json.dumps({"type": "keep-listening"}))
STOP_LISTENING = as_line(json.dumps({"type": "stop-listening"}))
AI_SPEECH = lambda chunk: as_line(
    json.dumps(
        {
            "type": "ai-speech",
            "sample_width": chunk.sample_width,
            "frame_rate": chunk.sample_rate,
            "channels": chunk.sample_channels,
            "data": base64.b64encode(chunk.audio_int16_bytes).decode("utf-8"),
        }
    )
)


def handle_microphone_ready_state(wakeword_model: openwakeword.model.Model):
    for chunk in microphone_chunks:
        wakeword_model.predict(np.frombuffer(chunk, dtype=np.int16))
        for model in wakeword_model.prediction_buffer.keys():
            score = list(wakeword_model.prediction_buffer[model])[-1]
            is_wakeword_detected = score > 0.5
            if is_wakeword_detected:
                microphone_chunks.clear()
                states["microphone"] = "pending"
                return


def handle_microphone_pending_state(max_silence_threshold_ms=3000):
    if len(microphone_chunks) == 0:
        return

    buf = b""
    for chunk in microphone_chunks:
        buf += chunk

    segment = pydub.AudioSegment.from_raw(
        io.BytesIO(buf), sample_width=2, frame_rate=16000, channels=1
    )

    ranges = pydub.silence.detect_silence(segment, min_silence_len=2000)
    max_silence = -1
    for start, end in ranges:
        if end - start > max_silence:
            max_silence = end - start

    if max_silence > max_silence_threshold_ms:
        buf = b""
        for chunk in microphone_chunks:
            buf += chunk
        microphone_chunks.clear()
        microphone_chunks.append(buf)
        states["microphone"] = "done"


def handle_microphone_done_state():
    microphone_chunks.clear()
    camera_frames.clear()
    states["microphone"] = "ready"


last_yielded_value = ""
last_mutation_id = ""
def ai_stream_generator(
    *,
    speech_client: SpeechClient,
    llm_client: LLMClient,
    wakeword_model: openwakeword.model.Model,
    stt_model: vosk.Model,
    chat_history: ChatHistory,
):
    global last_yielded_value
    global last_mutation_id

    while True:
        microphone_state = states["microphone"]

        if (
            last_yielded_value == STOP_LISTENING
            and len(microphone_chunks) > 0
            and len(camera_frames) > 0
        ):
            rec = vosk.KaldiRecognizer(stt_model, 16000)

            buf = b""
            for chunk in microphone_chunks:
                buf += chunk
            rec.AcceptWaveform(buf)
            prompt = json.loads(rec.FinalResult())['text']

            for chunk in speech_client.stream(
                text_stream=llm_client.stream(
                    model="gemma3:4b",
                    chat_history=chat_history.reset()
                    .add_system_message(SYSTEM_PROMPT)
                    .add_user_message(prompt, camera_frames),
                )
            ):
                yield AI_SPEECH(chunk)

            handle_microphone_done_state()
        else:
            match microphone_state:
                case "ready":
                    if last_mutation_id != microphone_chunks.mutation_id:
                        last_mutation_id = microphone_chunks.mutation_id
                        yield KEEP_LISTENING
                        handle_microphone_ready_state(wakeword_model)

                case "pending":
                    if last_mutation_id != microphone_chunks.mutation_id:
                        last_mutation_id = microphone_chunks.mutation_id
                        yield KEEP_LISTENING
                        handle_microphone_pending_state()

                case "done":
                    if last_yielded_value != STOP_LISTENING:
                        last_yielded_value = STOP_LISTENING
                        yield STOP_LISTENING
