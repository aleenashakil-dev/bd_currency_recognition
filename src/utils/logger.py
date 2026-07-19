"""Simple logging setup used across the project."""
import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name (usually __name__).
        level: One of DEBUG | INFO | WARNING | ERROR.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # Already configured
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger
