"""
utils/logger.py — Logging setup with rotating file handler + console output.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Bot started")
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from config import settings


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level. Defaults to INFO.

    Returns:
        Configured logging.Logger instance with console + file handlers.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — rotating, max 10 MB per file, keep 5 backups
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    log_file = os.path.join(settings.LOG_DIR, "bot.log")
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
