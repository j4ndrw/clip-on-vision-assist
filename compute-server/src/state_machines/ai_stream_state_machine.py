import json
from dataclasses import dataclass, field
from typing import Callable

from piper import AudioChunk
from src.camera_frames.camera_frames import camera_frames
from src.microphone_chunks.microphone_chunks import microphone_chunks
from src.state.state import State, states
from src.systems.ai_system import AISystem
from src.utils.stream import as_line, audio_chunk_as_line

@dataclass
class AIStreamStateMachineConfig:
    ai_system: AISystem
    microphone_idempotency_id_factory: Callable[[], str] = field(
        default=lambda: microphone_chunks.mutation_id
    )
    microphone_state_factory: Callable[[], State] = field(
        default=lambda: states["microphone"]
    )
    set_microphone_state: Callable[[State], None] = field(
        default=lambda state: states.update({"microphone": state})
    )


class AIStreamStateMachineSideEffects:
    def __init__(self, *, config: AIStreamStateMachineConfig):
        self.config = config

    def wakeword_detection(self):
        if self.config.ai_system.wakeword_detected(chunks=microphone_chunks):
            microphone_chunks.clear()
            self.config.set_microphone_state("pending")

    def listen(self):
        if len(microphone_chunks) > 0 and len(camera_frames) > 0:
            self.config.set_microphone_state("done")

    def reset_state(self):
        microphone_chunks.clear()
        camera_frames.clear()
        self.config.set_microphone_state("ready")

@dataclass
class AIStreamStateMachineEvent:
    CAPTURE_WAKEWORD: str = field(default=as_line(json.dumps({"type": "capture-wakeword"})))
    CAPTURE_PROMPT: str = field(default=as_line(json.dumps({"type": "capture-prompt"})))
    STALL: str = field(default=as_line(json.dumps({"type": "stall"})))
    AI_SPEECH: Callable[[AudioChunk], str] = field(default=audio_chunk_as_line("ai-speech"))

class AIStreamStateMachineEventProducer:
    def __init__(self, *, config: AIStreamStateMachineConfig):
        self.config = config

        self.stopped_listening_to_microphone = False
        self.requested_prompt = False

        self.last_microphone_mutation_id = ""

    def capture_wakeword(self):
        if (
            self.last_microphone_mutation_id
            != self.config.microphone_idempotency_id_factory()
        ):
            self.last_microphone_mutation_id = (
                self.config.microphone_idempotency_id_factory()
            )
            yield AIStreamStateMachineEvent.CAPTURE_WAKEWORD

    def capture_prompt(self):
        if not self.requested_prompt:
            self.requested_prompt = True
            yield AIStreamStateMachineEvent.CAPTURE_PROMPT

    def stall(self):
        if not self.stopped_listening_to_microphone:
            yield AIStreamStateMachineEvent.STALL
            self.stopped_listening_to_microphone = True

    def ai_speech(self):
        if len(microphone_chunks) > 0 and len(camera_frames) > 0:
            for speech in self.config.ai_system.stream_ai_speech(
                microphone_chunks=microphone_chunks, camera_frames=camera_frames
            ):
                yield AIStreamStateMachineEvent.AI_SPEECH(speech)

            self.stopped_listening_to_microphone = False
            self.requested_prompt = False


class AIStreamStateMachine:
    def __init__(self, *, config: AIStreamStateMachineConfig):
        self.config = config
        self.side_effects = AIStreamStateMachineSideEffects(config=config)
        self.event_producer = AIStreamStateMachineEventProducer(config=config)

    def generator(self):
        while True:
            match self.config.microphone_state_factory():
                case "ready":
                    for event in self.ready():
                        yield event

                case "pending":
                    for event in self.pending():
                        yield event

                case "done":
                    for event in self.done():
                        yield event

    def ready(self):
        for event in self.event_producer.capture_wakeword():
            yield event

        self.side_effects.wakeword_detection()

    def pending(self):
        if len(camera_frames) == 0:
            for event in self.event_producer.capture_prompt():
                yield event

        self.side_effects.listen()

    def done(self):
        for event in self.event_producer.stall():
            yield event

        for event in self.event_producer.ai_speech():
            yield event

        self.side_effects.reset_state()

def ai_stream_state_machine(*, config: AIStreamStateMachineConfig):
    state_machine = AIStreamStateMachine(config=config).generator()
    for event in state_machine:
        yield event
