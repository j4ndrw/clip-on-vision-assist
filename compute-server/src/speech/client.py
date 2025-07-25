from string import ascii_letters, digits
from typing import Any, Generator

import piper


class SpeechClient:
    _client: piper.PiperVoice = None  # pyright: ignore
    _synthesis_config: piper.SynthesisConfig = piper.SynthesisConfig(
        volume=0.5,
        noise_w_scale=1.0,
        normalize_audio=False,
    )

    def use(self, *, path: str):
        self._client = piper.PiperVoice.load(path)

    def set_synthesis_config(self, synthesis_config: piper.SynthesisConfig):
        self._synthesis_config = synthesis_config

    def get(self) -> piper.PiperVoice:
        return self._client

    def stream(self, *, text_stream: Generator[str, Any, Any]):
        chunks_to_synthesize: str = ""

        processed_chunks = 0
        for text_chunk in text_stream:
            has_punctuation = any(punctuation in text_chunk for punctuation in ".:?!")
            has_letter = any(letter in text_chunk for letter in ascii_letters)
            has_digits = any(digit in text_chunk for digit in digits)
            if has_punctuation and (has_letter or has_digits):
                chunks_to_synthesize += text_chunk
                for speech_chunk in self._client.synthesize(
                    chunks_to_synthesize, self._synthesis_config
                ):
                    yield speech_chunk
                chunks_to_synthesize = ""
                processed_chunks = 0
            else:
                chunks_to_synthesize += text_chunk
                processed_chunks += 1

        if len(chunks_to_synthesize) > 0:
            for speech_chunk in self._client.synthesize(
                chunks_to_synthesize, self._synthesis_config
            ):
                yield speech_chunk


speech_client = SpeechClient()
