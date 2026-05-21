from __future__ import annotations

import sys
import time
from typing import Any

_LEVEL_RANK = {"DEBUG": 10, "INFO": 20, "WARNING": 25, "ERROR": 30}
_level = "INFO"
_prefix = "[osm]"


def configure(level: str) -> None:
    """Set minimum log level for OSM preprocess messages."""
    global _level
    key = str(level).strip().upper()
    if key not in _LEVEL_RANK:
        raise ValueError(f"LOG_LEVEL must be DEBUG, INFO, WARNING, or ERROR (got {level!r})")
    _level = key


def debug_enabled() -> bool:
    """Return True when DEBUG logging is active."""
    return _LEVEL_RANK[_level] <= _LEVEL_RANK["DEBUG"]


def _enabled(level: str) -> bool:
    """Return True if messages at level should be emitted."""
    return _LEVEL_RANK[level] >= _LEVEL_RANK[_level]


def _emit(line: str, *, err: bool = False) -> None:
    """Print one prefixed line to stdout or stderr."""
    out = sys.stderr if err else sys.stdout
    print(f"{_prefix} {line}", file=out, flush=True)


def info(msg: str, *args: Any) -> None:
    """Log an informational pipeline message."""
    if not _enabled("INFO"):
        return
    text = (str(msg) % args) if args else str(msg)
    _emit(text)


def debug(msg: str, *args: Any) -> None:
    """Log a debug-only pipeline message."""
    if not _enabled("DEBUG"):
        return
    text = (str(msg) % args) if args else str(msg)
    _emit(text)


def warning(msg: str, *args: Any) -> None:
    """Log a non-fatal warning."""
    if not _enabled("WARNING"):
        return
    text = (str(msg) % args) if args else str(msg)
    _emit(text, err=True)


def error(msg: str) -> None:
    """Log a fatal or error message."""
    if _enabled("ERROR"):
        _emit(str(msg), err=True)


def sector_info(sector_id: str, msg: str, *args: Any) -> None:
    """Log a message prefixed with the sector id."""
    text = (str(msg) % args) if args else str(msg)
    info(f"[{sector_id}] {text}")


def sector_debug(sector_id: str, msg: str, *args: Any) -> None:
    """Log a sector-scoped debug message."""
    text = (str(msg) % args) if args else str(msg)
    debug(f"[{sector_id}] {text}")


def format_duration(seconds: float) -> str:
    """Format elapsed seconds as human-readable duration."""
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    m, r = divmod(int(round(s)), 60)
    return f"{m}m{r:02d}s"


def format_mib(path: Any) -> str:
    """Format file size in MiB from a path-like object."""
    from pathlib import Path

    p = Path(path)
    if not p.is_file():
        return "?"
    return f"{p.stat().st_size / (1024 * 1024):.1f} MiB"


class Timer:
    """Simple wall-clock timer for staged pipeline logging."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def elapsed(self) -> float:
        """Return seconds since timer creation."""
        return time.perf_counter() - self._t0
