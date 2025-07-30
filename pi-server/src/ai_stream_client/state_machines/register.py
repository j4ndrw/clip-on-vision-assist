from src.ai_stream_client.state_machines.state_machine import StateMachine
from src.ai_stream_client.state_machines.wakeword.state_machine import wakeword_based_state_machine

currently_active_state_machine: dict[str, StateMachine] = {"machine": wakeword_based_state_machine}
