"""Logging configuration using loguru."""

import sys

from loguru import logger

from kisan.core.config import get_settings


def setup_logging() -> None:
    """Configure loguru logger."""
    settings = get_settings()

    # Remove default handler
    logger.remove()

    # Add custom handler with formatting
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Add file handler for production
    if not settings.debug:
        logger.add(
            "logs/kisan_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
        )

    logger.info(f"Logging configured with level: {settings.log_level}")


# Export logger for use throughout the application
__all__ = ["logger", "setup_logging"]
