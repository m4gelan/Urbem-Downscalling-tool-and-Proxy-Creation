from __future__ import annotations

import inspect
import sys
from typing import Any
from datetime import datetime
from pathlib import Path

_LEVEL_RANK = {"DEBUG": 10, "INFO": 20, "WARNING": 25, "ERROR": 30}
_level = "INFO"

_GREEN = "\033[32m"
_ORANGE = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


def configure(level: str) -> None:
    global _level
    key = str(level).strip().upper()
    if key not in _LEVEL_RANK:
        raise ValueError(
            f"LOG_LEVEL must be DEBUG, INFO, WARNING, or ERROR (got {level!r})"
        )
    _level = key


def _enabled(level: str) -> bool:
    return _LEVEL_RANK[level] >= _LEVEL_RANK[_level]


def debug_enabled() -> bool:
    """True when LOG_LEVEL is DEBUG (enables debug-only outputs such as HTML maps)."""
    return _enabled("DEBUG")


def _caller() -> str:
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_back:
        fb = frame.f_back.f_back
        return f"{Path(fb.f_code.co_filename).name}:{fb.f_lineno} {fb.f_code.co_name}()"
    return "?:?"


def _emit(color: str, line: str, *, stream=None) -> None:
    out = sys.stderr if stream is sys.stderr else sys.stdout
    print(f"{color}{line}{_RESET}", file=out, flush=True)


def info(msg: str, *args: Any) -> None:
    if _enabled("INFO"):
        if args:
            try:
                text = str(msg) % args
            except (TypeError, ValueError):
                text = str(msg) + " " + " ".join(str(a) for a in args)
        else:
            text = str(msg)
        _emit(_GREEN, text)


def debug(msg: str, *args: Any) -> None:
    if _enabled("DEBUG"):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if args:
            try:
                text = str(msg) % args
            except (TypeError, ValueError):
                text = str(msg) + " " + " ".join(str(a) for a in args)
        else:
            text = str(msg)
        _emit(_ORANGE, f"{ts} [{_caller()}] {text}")


def warning(msg: str, *args: Any) -> None:
    """Non-fatal issues (shown at INFO, DEBUG, WARNING; hidden when LOG_LEVEL=ERROR)."""
    if not _enabled("WARNING"):
        return
    if args:
        try:
            text = str(msg) % args
        except (TypeError, ValueError):
            text = str(msg) + " " + " ".join(str(a) for a in args)
    else:
        text = str(msg)
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    _emit(_ORANGE, f"{ts} [{_caller()}] {text}", stream=sys.stderr)


def error(msg: str) -> None:
    if _enabled("ERROR"):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _emit(_RED, f"{ts} [{_caller()}] {msg}", stream=sys.stderr)
