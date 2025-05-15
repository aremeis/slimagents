# Global configuration variables
import logging

caching = False 

# The default logger for the slimagents package.
logger = logging.getLogger("slimagents")

# The root logger for the Agent class and subclasses (if separate_agent_logger is True).
# It defaults to the default logger (`slimagents`).
agent_logger = logger

# This option is used to specify if a separate logger should be created for each Agent subclass.
# If set to True, the logger will be created as a child of `agent_logger` with the name of the Agent subclass.
# If set to False, the default logger will be used.
# Default is False.
separate_agent_logger = False

# This option is used to specify if the streaming deltas should be logged. Only applicable for DEBUG level logging.
# Default is False.
debug_log_streaming_deltas = False

__all__ = ["caching", "logger", "agent_logger", "separate_agent_logger", "debug_log_streaming_deltas"]
