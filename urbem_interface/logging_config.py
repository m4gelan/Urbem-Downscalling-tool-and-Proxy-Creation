"""
UrbEm Interface logging configuration.
"""

import logging
import sys


GREEN = "\033[92m"
ORANGE = "\033[38;5;208m"
RED = "\033[91m"
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors based on log level."""

    LEVEL_COLORS = {
        logging.INFO: GREEN,
        logging.DEBUG: ORANGE,
        logging.ERROR: RED,
        logging.WARNING: ORANGE,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        record.msg = f"{color}{record.msg}{RESET}"
        return super().format(record)


def get_logger(name: str, level: str | int | None = None) -> logging.Logger:
    """Get a logger for the urbem_interface package."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(level or _resolve_log_level())
    logger.propagate = False
    return logger


def _resolve_log_level() -> int:
    import os
    level_str = os.environ.get("URBEM_LOG_LEVEL", "INFO").upper()
    return logging.DEBUG if level_str == "DEBUG" else logging.INFO


def configure_urbem_logging(debug: bool | None = None) -> None:
    """Configure logging for the urbem_interface package."""
    import os
    if debug is None:
        debug = os.environ.get("URBEM_LOG_LEVEL", "INFO").upper() == "DEBUG"
    level = logging.DEBUG if debug else logging.INFO
    os.environ["URBEM_LOG_LEVEL"] = "DEBUG" if debug else "INFO"
    for name in (
        "urbem_interface",
        "urbem_interface.emissions",
        "urbem_interface.emissions.area_sources",
        "urbem_interface.emissions.prepare_cams",
        "urbem_interface.emissions.proxy_preparation",
        "urbem_interface.utils.grid",
        "urbem_interface.utils.grid.grid_warp",
    ):
        logging.getLogger(name).setLevel(level)
