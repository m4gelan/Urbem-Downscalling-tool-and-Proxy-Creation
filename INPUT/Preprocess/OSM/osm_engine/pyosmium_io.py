from __future__ import annotations

from pathlib import Path
from typing import Any

from . import log

# flex_mem is always available in conda pyosmium; dense_mmap often is not on Windows builds.
_IDX_FALLBACK = "flex_mem"


def pick_pyosmium_idx(
    work_pbf: Path,
    defaults: dict[str, Any],
    sector_entry: dict[str, Any],
) -> str:
    """Choose location index; use flex_mem for small/filtered PBF (safe on all builds)."""
    if sector_entry.get("pyosmium_idx"):
        return str(sector_entry["pyosmium_idx"])
    if work_pbf.is_file() and work_pbf.stat().st_size <= 150 * 1024 * 1024:
        return _IDX_FALLBACK
    if defaults.get("pyosmium_idx"):
        return str(defaults["pyosmium_idx"])
    return _IDX_FALLBACK


def apply_file(
    handler: Any,
    work_pbf: Path,
    *,
    sector_id: str,
    idx: str,
) -> str:
    """Run pyosmium apply_file; fall back to flex_mem if idx type is not compiled in."""
    path = str(work_pbf)
    try:
        handler.apply_file(path, locations=True, idx=idx)
        return idx
    except RuntimeError as e:
        msg = str(e).lower()
        if idx != _IDX_FALLBACK and ("not compiled" in msg or "map type" in msg):
            log.warning(
                f"[{sector_id}] idx={idx} unavailable in this pyosmium build, using {_IDX_FALLBACK}"
            )
            handler.apply_file(path, locations=True, idx=_IDX_FALLBACK)
            return _IDX_FALLBACK
        raise
