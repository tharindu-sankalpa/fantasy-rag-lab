"""Centralized Logging Configuration.

This module configures `structlog` to provide structured logging that adapts to the 
execution environment.

Behavior:
- **Local Development (Interactive TTY)**: Renders colored, human-readable console output 
  with simple timestamps (%H:%M:%S).
- **Production (K8s/Docker/Non-Interactive)**: Renders machine-parseable JSON output 
  with ISO 8601 timestamps, ideal for log aggregators (ELK, Splunk, Datadog).

Configuration details:
- **merge_contextvars**: Merges global context variables (e.g., request IDs) into the event dict.
- **add_log_level**: Adds the log level (INFO, ERROR, etc.) to the event dict.
- **StackInfoRenderer**: Adds stack info if `stack_info=True` is passed.
- **set_exc_info**: Automatically adds exception info if `exc_info` is present or if logging within an exception handler.
- **TimeStamper**: Environment-aware timestamp format.
- **Renderer**: Auto-switches between `ConsoleRenderer` and `JSONRenderer`.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("event_name", key="value")
"""

import structlog
import logging
import sys

# Check if running in an interactive terminal or a non-interactive environment (like K8s/Docker)
# If stderr is a TTY, we are likely developing locally -> use pretty colors.
# If not, we are likely piping to a file or running in K8s -> use JSON for ELK/Splunk/Datadog.
min_level = logging.INFO
renderer = (
    structlog.dev.ConsoleRenderer(colors=True)
    if sys.stderr.isatty()
    else structlog.processors.JSONRenderer()
)

# For JSON logs, we usually want a standard ISO timestamp, not just simple time.
timestamper = (
    structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False)
    if sys.stderr.isatty()
    else structlog.processors.TimeStamper(fmt="iso")
)

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        timestamper,
        renderer,
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

def get_logger(name=None):
    """Retrieves a structlog logger instance.

    This function is a wrapper around `structlog.get_logger` to ensure consistent 
    logger acquisition throughout the application.

    Args:
        name: Optional name for the logger, typically `__name__` of the calling module. 
              Useful for filtering or identifying the source of logs.

    Returns:
        A structlog bound logger instance configured with the project's processors.
    """
    return structlog.get_logger(name)

