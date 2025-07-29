from enum import Enum, auto


class StreamEventType(Enum):
    CAPTURE_WAKEWORD = "capture-wakeword"
    CAPTURE_PROMPT = "capture-prompt"
    STALL = "stall"
    AI_SPEECH = "ai-speech"
    UNKNOWN = auto()
