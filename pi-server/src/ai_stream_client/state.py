from dataclasses import dataclass, field
from typing import Callable

from src.ai_stream_client.event import StreamEventType


@dataclass
class State:
    type: StreamEventType | None = field(default=None)
    task: Callable[[], None] | None = field(default=None)
