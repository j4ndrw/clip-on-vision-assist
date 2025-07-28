import io
import json

import numpy as np
import openwakeword.model
import pydub
import pydub.silence
import vosk
from openai.types.shared.chat_model import ChatModel
from src.llm.client import LLMClient
from src.llm.history import ChatHistory
from src.llm.prompts import SYSTEM_PROMPT
from src.speech.client import SpeechClient


class AISystem:
    def __init__(
        self,
        *,
        speech_client: SpeechClient,
        llm_client: LLMClient,
        wakeword_model: openwakeword.model.Model,
        stt_model: vosk.Model,
        chat_history: ChatHistory,
        llm: ChatModel | str,
    ):
        self.speech_client = speech_client
        self.llm_client = llm_client
        self.wakeword_model = wakeword_model
        self.stt_model = stt_model
        self.chat_history = chat_history
        self.llm = llm

    def wakeword_detected(
        self,
        *,
        chunks: list[bytes],
        wakeword_score_threshold=0.8,
    ) -> bool:
        for chunk in chunks:
            self.wakeword_model.predict(np.frombuffer(chunk, dtype=np.int16))
            for model in self.wakeword_model.prediction_buffer.keys():
                score = list(self.wakeword_model.prediction_buffer[model])[-1]
                is_wakeword_detected = score > wakeword_score_threshold
                if is_wakeword_detected:
                    return True

        return False

    def get_audio_chunks_until_silent(
        self, *, audio_chunks: list[bytes], max_silence_threshold_ms=3000
    ) -> bytes | None:
        buf = b"".join(audio_chunks)

        segment = pydub.AudioSegment.from_raw(
            io.BytesIO(buf), sample_width=2, frame_rate=16000, channels=1
        )

        silent_ending_index = pydub.silence.detect_leading_silence(segment)
        ranges = pydub.silence.detect_silence(segment, min_silence_len=2000)
        max_silence = -1
        for start, end in ranges:
            if start > silent_ending_index and end - start > max_silence:
                max_silence = end - start

        if max_silence <= max_silence_threshold_ms:
            return None

        return buf

    def stream_ai_speech(
        self, *, microphone_chunks: list[bytes], camera_frames: list[str]
    ):
        rec = vosk.KaldiRecognizer(self.stt_model, 16000)

        buf = b""
        for chunk in microphone_chunks:
            buf += chunk
        rec.AcceptWaveform(buf)
        prompt = json.loads(rec.FinalResult())["text"]

        llm_text_stream = self.llm_client.stream(
            model=self.llm,
            chat_history=self.chat_history.reset()
            .add_system_message(SYSTEM_PROMPT)
            .add_user_message(prompt, camera_frames),
        )
        for chunk in self.speech_client.stream(text_stream=llm_text_stream):
            yield chunk
