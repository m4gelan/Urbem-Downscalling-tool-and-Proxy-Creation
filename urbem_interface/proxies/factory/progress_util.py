"""Optional tqdm-based progress (install: pip install tqdm)."""

from __future__ import annotations

import sys
import threading
import time
from typing import Any, Callable, Iterable, TypeVar

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:
    _tqdm_cls = None

T = TypeVar("T")
R = TypeVar("R")


def tqdm_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    enabled: bool = True,
    **kwargs: Any,
) -> Iterable[T]:
    if not enabled or _tqdm_cls is None:
        return iterable
    return _tqdm_cls(iterable, desc=desc, total=total, unit=unit, **kwargs)


def tqdm_available() -> bool:
    return _tqdm_cls is not None


def run_with_pulse_progress(
    fn: Callable[[], R],
    *,
    desc: str,
    enabled: bool = True,
    poll_s: float = 0.25,
) -> R:
    """
    Run fn() in a worker thread; main thread sleeps and ticks tqdm.

    If fn() runs on the main thread, the GIL can stay with GDAL/pyogrio for the whole read,
    so a helper thread never gets to call pbar.update() (bar stuck at 0). The worker thread
    usually releases the GIL during native I/O, so the main thread can keep updating.
    """
    if not enabled or _tqdm_cls is None:
        return fn()

    result: list[R] = []
    error: list[BaseException] = []

    def target() -> None:
        try:
            result.append(fn())
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=target, daemon=True)
    with _tqdm_cls(
        total=None,
        desc=desc,
        unit="pulse",
        mininterval=0,
        file=sys.stderr,
        bar_format="{desc} | {n_fmt} pulses | {elapsed} elapsed",
        dynamic_ncols=True,
    ) as pbar:
        thread.start()
        pbar.refresh()
        sys.stderr.flush()
        while thread.is_alive():
            time.sleep(poll_s)
            pbar.update(1)
            pbar.refresh()
            sys.stderr.flush()
        thread.join()
    if error:
        raise error[0]
    return result[0]
