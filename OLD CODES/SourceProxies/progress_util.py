"""Optional tqdm progress bars (install ``tqdm`` for bars; otherwise no-op)."""

from __future__ import annotations

import sys
from typing import Any, Iterable, TypeVar

T = TypeVar("T")


def tqdm_if_installed(
    iterable: Iterable[T],
    *,
    desc: str,
    unit: str = "it",
    total: int | None = None,
    file: Any = None,
) -> Iterable[T]:
    out_file = file if file is not None else sys.stderr
    try:
        from tqdm import tqdm

        kw: dict[str, Any] = {"desc": desc, "unit": unit, "file": out_file}
        if total is not None:
            kw["total"] = total
        return tqdm(iterable, **kw)
    except ImportError:
        return iterable


def note(msg: str, *, file: Any = None) -> None:
    print(msg, file=file if file is not None else sys.stderr, flush=True)
