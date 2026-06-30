"""Lightweight logging setup for ShearNet.

Provides a package logger (``shearnet``) wired to stdout with a plain,
message-only format, so existing ``print``-style status output is preserved
while becoming controllable: callers can raise/lower the level, redirect the
stream, or attach their own handlers.

Usage in a module::

    from .logging_utils import get_logger

    logger = get_logger(__name__)
    logger.info("hello")

Applications can adjust verbosity, e.g.::

    import logging
    from shearnet.logging_utils import configure_logging
    configure_logging(level=logging.WARNING)   # quieter
"""

import logging
import os
import sys

_LOGGER_NAME = "shearnet"


def supports_color(stream=None) -> bool:
    """Whether ANSI color codes should be emitted to ``stream``.

    Honors the ``NO_COLOR`` / ``FORCE_COLOR`` conventions, otherwise enables
    color only when the stream is an interactive terminal.
    """
    stream = stream if stream is not None else sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


def ansi(code: str, stream=None) -> str:
    """Return the ANSI ``code`` if the stream supports color, else ``""``.

    Lets modules define color constants (``BOLD = ansi("\\033[1m")``) that simply
    vanish when output is piped or redirected, instead of writing escape codes
    into log files.
    """
    return code if supports_color(stream) else ""


def configure_logging(level=logging.INFO, stream=None, force=False):
    """Configure the ``shearnet`` logger (idempotent).

    Attaches a single stdout :class:`logging.StreamHandler` with a
    message-only formatter, so output matches the previous ``print`` behavior.

    Args:
        level: Logging level for the package logger.
        stream: Output stream (defaults to ``sys.stdout``).
        force: Replace any existing handlers when ``True``.

    Returns:
        logging.Logger: The configured ``shearnet`` logger.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
    if not logger.handlers:
        handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        # Don't double-emit through the root logger if the app configured it.
        logger.propagate = False
    logger.setLevel(level)
    return logger


def get_logger(name=None):
    """Return the ``shearnet`` logger or a child of it.

    Pass ``__name__`` to get a per-module child logger (e.g.
    ``shearnet.core.train``); all children propagate to the package logger's
    handler.
    """
    if not name or name == _LOGGER_NAME:
        return logging.getLogger(_LOGGER_NAME)
    if not name.startswith(_LOGGER_NAME + "."):
        name = f"{_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


# Configure on import so library output is visible by default (preserving the
# previous print-to-stdout behavior) while remaining user-overridable.
configure_logging()
