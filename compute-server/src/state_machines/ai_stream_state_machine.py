import base64
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

from src.camera_frames.camera_frames import camera_frames
from src.microphone_chunks.microphone_chunks import microphone_chunks
from src.state.state import State, states
from src.systems.ai_system import AISystem
from src.utils.stream import as_line


@dataclass
class AIStreamStateMachineConfig:
    ai_system: AISystem
    microphone_idempotency_id_factory: Callable[[], str] = field(
        default=lambda: microphone_chunks.mutation_id
    )
    camera_idempotency_id_factory: Callable[[], str] = field(
        default=lambda: camera_frames.mutation_id
    )
    microphone_state_factory: Callable[[], State] = field(
        default=lambda: states["microphone"]
    )
    set_microphone_state: Callable[[State], None] = field(
        default=lambda state: states.update({"microphone": state})
    )


class AIStreamStateMachine:
    LISTEN = as_line(json.dumps({"type": "listen"}))
    TAKE_PICTURES_AND_LISTEN = as_line(json.dumps({"type": "take-pictures-and-listen"}))
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

    def __init__(self, *, config: AIStreamStateMachineConfig):
        self.system = config.ai_system

        self.microphone_state_factory = config.microphone_state_factory
        self.set_microphone_state = config.set_microphone_state
        self.microphone_idempotency_id_factory = config.microphone_idempotency_id_factory
        self.camera_idempotency_id_factory = config.camera_idempotency_id_factory

        self.stopped_listening_to_microphone = False

        self.last_microphone_mutation_id = ""
        self.last_camera_mutation_id = ""

    def execute(self):
        while True:
            match self.microphone_state_factory():
                case "ready":
                    for event in self.with_idempotency_check(self.ready()):
                        yield event

                case "pending":
                    for event in self.with_idempotency_check(self.pending()):
                        yield event

                case "done":
                    if not self.stopped_listening_to_microphone:
                        yield AIStreamStateMachine.STOP_LISTENING
                        self.stopped_listening_to_microphone = True
                    elif (
                        len(microphone_chunks) > 0
                        and len(camera_frames) > 0
                    ):
                        for event in self.done():
                            yield event
                        self.stopped_listening_to_microphone = False

    def with_idempotency_check(self, generator: Generator[str, Any, None]):
        if self.last_microphone_mutation_id == self.microphone_idempotency_id_factory():
            return

        if self.last_camera_mutation_id == self.camera_idempotency_id_factory():
            return

        self.last_microphone_mutation_id = self.microphone_idempotency_id_factory()
        self.last_camera_mutation_id = self.camera_idempotency_id_factory()

        for value in generator:
            yield value

    def ready(self):
        yield AIStreamStateMachine.LISTEN
        if len(camera_frames) > 0 or (
            len(camera_frames) == 0
            and self.system.wakeword_detected(chunks=microphone_chunks)
        ):
            microphone_chunks.clear()
            self.set_microphone_state("pending")

    def pending(self):
        if len(camera_frames) == 0:
            yield AIStreamStateMachine.TAKE_PICTURES_AND_LISTEN
        else:
            yield AIStreamStateMachine.LISTEN

        if len(microphone_chunks) > 0 and len(camera_frames) > 0:
            if buf := self.system.get_audio_chunks_until_silent(
                audio_chunks=microphone_chunks
            ):
                microphone_chunks.clear()
                microphone_chunks.append(buf)
                self.set_microphone_state("done")

    def done(self):
        for speech in self.system.stream_ai_speech(
            microphone_chunks=microphone_chunks, camera_frames=camera_frames
        ):
            yield AIStreamStateMachine.AI_SPEECH(speech)

        microphone_chunks.clear()
        camera_frames.clear()
        self.set_microphone_state("ready")


def ai_stream_state_machine(*, config: AIStreamStateMachineConfig):
    state_machine = AIStreamStateMachine(config=config)
    for event in state_machine.execute():
        yield event
