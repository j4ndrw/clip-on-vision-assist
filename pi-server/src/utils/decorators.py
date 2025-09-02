import asyncio
import time
from functools import wraps
from typing import Any, Callable, Coroutine

from src.env import environment


def with_interrupt(delay: float = 0.2):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            time.sleep(delay)
            return result

        return wrapper

    return decorator


def with_interrupt_async(delay: float = 0.2):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            await asyncio.sleep(delay)
            return result

        return wrapper

    return decorator


class LoopBreak:
    @staticmethod
    def check(x: Any):
        return isinstance(x, LoopBreak)


def looped(func: Callable[[], bool]):
    @wraps(func)
    def wrapper(*args, **kwargs):
        while not LoopBreak.check(func(*args, **kwargs)):
            pass

    return wrapper


def looped_async(func: Callable[[], Coroutine[Any, Any, bool]]):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        while not LoopBreak.check(await func(*args, **kwargs)):
            pass

    return wrapper


def production_only(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        env = environment.get().get("ENVIRONMENT") or "local"
        if env == "local":
            return None
        return func(*args, **kwargs)

    return wrapper


def production_only_async(func: Callable):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        env = environment.get().get("ENVIRONMENT") or "local"
        if env == "local":
            return
        await func(*args, **kwargs)

    return wrapper
