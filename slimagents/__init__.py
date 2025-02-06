from .core import Agent, Response, Result, logger
from .repl import run_demo_loop, run_demo_loop_async
from importlib.metadata import version

try:
    __version__ = version("slimagents")
except Exception:
    __version__ = "unknown"
    
logger.name = __name__

__all__ = ["Agent", "Response", "Result", "run_demo_loop", "run_demo_loop_async", "logger"]
