import base64
import json
from dataclasses import dataclass, field
from typing import Callable

from piper import AudioChunk
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
        if len(camera_frames) > 0 or (
            len(camera_frames) == 0
            and self.config.ai_system.wakeword_detected(chunks=microphone_chunks)
        ):
            microphone_chunks.clear()
            self.config.set_microphone_state("pending")

    def listen_until_silent(self):
        if len(microphone_chunks) > 0 and len(camera_frames) > 0:
            if buf := self.config.ai_system.get_audio_chunks_until_silent(
                audio_chunks=microphone_chunks
            ):
                microphone_chunks.clear()
                microphone_chunks.append(buf)
                self.config.set_microphone_state("done")

    def reset_state(self):
        microphone_chunks.clear()
        camera_frames.clear()
        self.config.set_microphone_state("ready")

@dataclass
class AIStreamStateMachineEvent:
    LISTEN: str = field(default=as_line(json.dumps({"type": "listen"})))
    TAKE_PICTURES: str = field(default=as_line(json.dumps({"type": "take-pictures"})))
    STOP_LISTENING: str = field(default=as_line(json.dumps({"type": "stop-listening"})))
    AI_SPEECH: Callable[[AudioChunk], str] = field(default=lambda chunk: as_line(
        json.dumps(
            {
                "type": "ai-speech",
                "sample_width": chunk.sample_width,
                "frame_rate": chunk.sample_rate,
                "channels": chunk.sample_channels,
                "data": base64.b64encode(chunk.audio_int16_bytes).decode("utf-8"),
            }
        )
    ))

class AIStreamStateMachineEventProducer:
    def __init__(self, *, config: AIStreamStateMachineConfig):
        self.config = config

        self.stopped_listening_to_microphone = False
        self.requested_pictures = False

        self.last_microphone_mutation_id = ""

    def listen(self):
        if (
            self.last_microphone_mutation_id
            != self.config.microphone_idempotency_id_factory()
        ):
            self.last_microphone_mutation_id = (
                self.config.microphone_idempotency_id_factory()
            )
            yield AIStreamStateMachineEvent.LISTEN

    def take_pictures(self):
        if not self.requested_pictures:
            self.requested_pictures = True
            yield AIStreamStateMachineEvent.TAKE_PICTURES

    def stop_listening(self):
        if not self.stopped_listening_to_microphone:
            yield AIStreamStateMachineEvent.STOP_LISTENING
            self.stopped_listening_to_microphone = True

    def ai_speech(self):
        if len(microphone_chunks) > 0 and len(camera_frames) > 0:
            for speech in self.config.ai_system.stream_ai_speech(
                microphone_chunks=microphone_chunks, camera_frames=camera_frames
            ):
                yield AIStreamStateMachineEvent.AI_SPEECH(speech)

            self.stopped_listening_to_microphone = False
            self.requested_pictures = False

class AIStreamStateMachine:
    def __init__(self, *, config: AIStreamStateMachineConfig):
        self.config = config
        self.side_effects = AIStreamStateMachineSideEffects(config=config)
        self.event_producer = AIStreamStateMachineEventProducer(config=config)

    def execute(self):
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
        for event in self.event_producer.listen():
            yield event

        self.side_effects.wakeword_detection()

    def pending(self):
        if len(camera_frames) == 0:
            for event in self.event_producer.take_pictures():
                yield event

        for event in self.event_producer.listen():
            yield event

        self.side_effects.listen_until_silent()

    def done(self):
        for event in self.event_producer.stop_listening():
            yield event

        for event in self.event_producer.ai_speech():
            yield event

        self.side_effects.reset_state()


def ai_stream_state_machine(*, config: AIStreamStateMachineConfig):
    state_machine = AIStreamStateMachine(config=config)
    for event in state_machine.execute():
        yield event
