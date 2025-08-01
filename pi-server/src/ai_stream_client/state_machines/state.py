from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generic, TypeVar, Optional

T = TypeVar("T", bound=Enum)

@dataclass
class State(Generic[T]):
    type: Optional[T] = field(default=None)
    task: Optional[Callable[[], None]] = field(default=None)
