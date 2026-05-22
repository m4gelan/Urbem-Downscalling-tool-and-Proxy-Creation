from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import osmium

from . import log

_TICK_EVERY = 50_000
_LOG_EVERY = 500_000


def _extract_object_total_from_fileinfo_json(data: Any) -> int | None:
    """Best-effort nodes+ways+relations count from osmium fileinfo -e -j JSON."""
    if isinstance(data, dict):
        if all(k in data for k in ("nodes", "ways", "relations")):
            try:
                return int(data["nodes"]) + int(data["ways"]) + int(data["relations"])
            except (TypeError, ValueError):
                pass
        for v in data.values():
            t = _extract_object_total_from_fileinfo_json(v)
            if t is not None:
                return t
    if isinstance(data, list):
        for x in data:
            t = _extract_object_total_from_fileinfo_json(x)
            if t is not None:
                return t
    return None


def estimate_pbf_objects(osmium_exe: str | None, work_pbf: Path) -> int | None:
    """Return approximate OSM object count for tqdm total (slow on huge PBF)."""
    if not osmium_exe or not work_pbf.is_file():
        return None
    try:
        r = subprocess.run(
            [osmium_exe, "fileinfo", "-e", "-j", str(work_pbf)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if r.returncode != 0 or not (r.stdout or "").strip():
            return None
        data = json.loads(r.stdout)
        return _extract_object_total_from_fileinfo_json(data)
    except (json.JSONDecodeError, subprocess.TimeoutExpired, OSError, ValueError):
        return None


class ProgressWrapper(osmium.SimpleHandler):
    """SimpleHandler shell that forwards callbacks to an inner handler and counts progress."""

    def __init__(
        self,
        handler: Any,
        *,
        sector_id: str,
        pbar: Any | None = None,
        tick_every: int = _TICK_EVERY,
        log_every: int = _LOG_EVERY,
    ) -> None:
        super().__init__()
        self._h = handler
        self._sector_id = sector_id
        self._pbar = pbar
        self._tick_every = tick_every
        self._log_every = log_every
        self.objects_seen = 0

    def _tick(self) -> None:
        """Count one OSM object and update bar or log."""
        self.objects_seen += 1
        if self._pbar is not None:
            if self.objects_seen % self._tick_every == 0:
                self._pbar.update(self._tick_every)
        elif self.objects_seen % self._log_every == 0:
            log.sector_info(self._sector_id, f"parse scanned {self.objects_seen:,} objects ...")

    def node(self, n: Any) -> None:
        """Forward node callback with progress tick."""
        self._tick()
        self._h.node(n)

    def way(self, w: Any) -> None:
        """Forward way callback with progress tick."""
        self._tick()
        self._h.way(w)

    def area(self, a: Any) -> None:
        """Forward area callback with progress tick."""
        self._tick()
        self._h.area(a)

    def relation(self, r: Any) -> None:
        """Count relations for progress (handler may not implement relation)."""
        self._tick()
        if hasattr(self._h, "relation"):
            self._h.relation(r)

    def close_progress(self) -> None:
        """Flush remaining tqdm increments and close the bar."""
        if self._pbar is None:
            return
        rem = self.objects_seen % self._tick_every
        if rem:
            self._pbar.update(rem)
        self._pbar.close()


def wrap_handler(
    handler: Any,
    *,
    sector_id: str,
    work_pbf: Path,
    osmium_exe: str | None,
    show_progress: bool,
) -> tuple[Any, ProgressWrapper | None]:
    """Return handler (possibly wrapped) and wrapper for close_progress."""
    if not show_progress:
        return handler, None

    pbar = None
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]

        total = estimate_pbf_objects(osmium_exe, work_pbf)
        pbar = tqdm(
            total=total,
            unit="obj",
            desc=f"[{sector_id}] pyosmium",
            smoothing=0.05,
            mininterval=0.5,
        )
        if total is None:
            log.sector_info(sector_id, "parse progress (no total; fileinfo skipped or failed)")
    except ImportError:
        log.sector_info(
            sector_id,
            "parse progress: install tqdm for a bar (pip install tqdm), else log every "
            f"{_LOG_EVERY:,} objects",
        )

    wrap = ProgressWrapper(handler, sector_id=sector_id, pbar=pbar)
    return wrap, wrap
