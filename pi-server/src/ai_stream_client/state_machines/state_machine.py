from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

from src.ai_stream_client.client import AIStreamClient
from src.ai_stream_client.state_machines.state import State

TState = TypeVar("TState", bound=Enum)


@dataclass
class StateMachineConfig(Generic[TState]):
    client: AIStreamClient
    state: State[TState]
    msg: Any


StateMachine = Callable[[StateMachineConfig[TState]], State[TState]]
