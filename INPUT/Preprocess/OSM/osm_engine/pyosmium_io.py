from __future__ import annotations

from pathlib import Path
from typing import Any

from . import log
from . import parse_progress

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


def _show_parse_progress(sector_entry: dict[str, Any], defaults: dict[str, Any]) -> bool:
    """Return True unless progress is explicitly disabled in config."""
    if sector_entry.get("show_parse_progress") is False:
        return False
    if defaults.get("show_parse_progress") is False:
        return False
    return sector_entry.get("show_parse_progress", defaults.get("show_parse_progress", True)) is not False


def _apply_raw(handler: Any, path: str, idx: str, *, sector_id: str) -> str:
    """Run pyosmium apply_file once with the given index type."""
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


def apply_file(
    handler: Any,
    work_pbf: Path,
    *,
    sector_id: str,
    idx: str,
    osmium_exe: str | None = None,
    defaults: dict[str, Any] | None = None,
    sector_entry: dict[str, Any] | None = None,
) -> str:
    """Run pyosmium apply_file with optional tqdm / periodic progress."""
    defaults = defaults or {}
    sector_entry = sector_entry or {}
    path = str(work_pbf)
    show = _show_parse_progress(sector_entry, defaults)
    target, prog = parse_progress.wrap_handler(
        handler,
        sector_id=sector_id,
        work_pbf=work_pbf,
        osmium_exe=osmium_exe,
        show_progress=show,
    )
    try:
        return _apply_raw(target, path, idx, sector_id=sector_id)
    finally:
        if prog is not None:
            prog.close_progress()
