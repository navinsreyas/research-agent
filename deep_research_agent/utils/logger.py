"""
Logging setup for the Deep Research Agent.

Configures a rotating file handler that writes detailed execution logs to
logs/deep_research_agent.log without polluting the user-facing CLI output.
Console output is reserved for user-facing progress messages in run_agent.py.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_file: str = "deep_research_agent.log"
) -> logging.Logger:
    """
    Create a logger with a rotating file handler.

    Rotation: 5 MB per file, 3 backup files kept (15 MB total max).
    Level: INFO for file output; DEBUG available by changing file_handler.setLevel.

    Args:
        name:     Logger name — typically __name__ from the calling module.
        log_dir:  Directory for log files (created automatically if absent).
        log_file: Log filename.

    Returns:
        Configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called more than once
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    log_path = os.path.join(log_dir, log_file)
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
