from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar

T = TypeVar("T", bound=Enum)

@dataclass
class State(Generic[T]):
    type: T | None = field(default=None)
    task: Callable[[], None] | None = field(default=None)
