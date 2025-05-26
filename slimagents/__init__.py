from .core import Agent, Response, ToolResult, MessageDelimiter, DelimiterType
from .repl import run_demo_loop, run_demo_loop_async
from importlib.metadata import version
from .config import logger

try:
    __version__ = version("slimagents")
except Exception:
    __version__ = "unknown"
    
logger.name = __name__

__all__ = ["Agent", "Response", "ToolResult", "MessageDelimiter", "DelimiterType", "run_demo_loop", "run_demo_loop_async", "logger"]
