from enum import Enum, auto


class WakewordBasedStreamEventType(Enum):
    CAPTURE_WAKEWORD = "capture-wakeword"
    CAPTURE_PROMPT = "capture-prompt"
    STALL = "stall"
    AI_SPEECH = "ai-speech"
    DONE = "done"
    UNKNOWN = auto()
