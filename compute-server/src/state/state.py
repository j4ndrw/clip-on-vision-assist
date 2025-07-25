from typing import Literal

State = Literal["ready"] | Literal["pending"] | Literal["done"]
StateKey = Literal["microphone"]

states: dict[StateKey, State] = {"microphone": "ready"}
