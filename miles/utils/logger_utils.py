"""Named loggers safe to import before initializing training dependencies."""

from logging import Logger, getLogger


def setup_logger(name: str) -> Logger:
    """Return a named logger using the worker's existing logging configuration."""
    return getLogger(name)
