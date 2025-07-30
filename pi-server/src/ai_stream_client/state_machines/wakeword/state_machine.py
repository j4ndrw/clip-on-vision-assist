from src.ai_stream_client.state_machines.wakeword.event import WakewordBasedStreamEventType
from src.ai_stream_client.state_machines.state_machine import StateMachineConfig
from src.ai_stream_client.state_machines.wakeword.tasks import WakewordBasedStateMachineTasks

def wakeword_based_state_machine(config: StateMachineConfig[WakewordBasedStreamEventType]):
    config.state.type = WakewordBasedStreamEventType(config.msg["type"])
    tasks = WakewordBasedStateMachineTasks(client=config.client)
    match config.state.type:
        case WakewordBasedStreamEventType.CAPTURE_WAKEWORD:
            config.state.task = tasks.capture_wakeword()
        case WakewordBasedStreamEventType.CAPTURE_PROMPT:
            config.state.task = tasks.capture_prompt()
        case WakewordBasedStreamEventType.STALL:
            config.state.task = tasks.stall()
        case WakewordBasedStreamEventType.AI_SPEECH:
            config.state.task = tasks.ai_speech(config.msg)
        case WakewordBasedStreamEventType.DONE:
            config.state.task = tasks.done()

    return config.state
