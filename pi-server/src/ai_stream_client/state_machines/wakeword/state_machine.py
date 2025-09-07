from src.ai_stream_client.state_machines.state_machine import StateMachineConfig
from src.ai_stream_client.state_machines.wakeword.event import (
    WakewordBasedStreamEventType,
)
from src.ai_stream_client.state_machines.wakeword.tasks import (
    WakewordBasedStateMachineTasks,
)


def wakeword_based_state_machine(
    config: StateMachineConfig[WakewordBasedStreamEventType],
):
    config.state.type = WakewordBasedStreamEventType(config.msg["type"])
    tasks = WakewordBasedStateMachineTasks(client=config.client)
    switch = {
        WakewordBasedStreamEventType.CAPTURE_WAKEWORD: lambda: tasks.capture_wakeword(),
        WakewordBasedStreamEventType.CAPTURE_PROMPT: lambda: tasks.capture_prompt(),
        WakewordBasedStreamEventType.STALL: lambda: tasks.stall(),
        WakewordBasedStreamEventType.AI_SPEECH: lambda: tasks.ai_speech(config.msg),
        WakewordBasedStreamEventType.DONE: lambda: tasks.done(),
    }

    task_factory = switch.get(config.state.type)
    if task_factory is not None:
        config.state.task = task_factory()
    return config.state
