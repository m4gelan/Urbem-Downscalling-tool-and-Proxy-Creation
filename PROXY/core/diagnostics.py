"""Shared diagnostic helpers for sector pipelines.

Phase 3.1 extraction. Previously ``B_Industry/pipeline.py`` and
``D_Fugitive/pipeline.py`` each carried a near-identical copy of:

- ``_max_valid_cam_cell_id`` (chunked maximum of valid CAMS cell IDs);
- ``_log_raster_stats`` (shape / finite / min / max / mean line);
- ``_log_population_chain`` (population -> density -> P_pop stats);
- ``_log_cam_and_country`` (CAMS-cell and NUTS-country coverage).

These now live here behind a small stateful :class:`RasterStatsLogger` that carries the
sector tag (``[industry-debug]`` / ``[fugitive-debug]`` etc.) and a logger instance so
formatting stays faithful to the original messages.

The functions stay behaviourally identical to the originals; the only change is that
callers pass their logger and tag instead of relying on module globals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

_DEFAULT_CHUNK = 8_000_000


def max_valid_cam_cell_id(cam_cell_id: np.ndarray, chunk_elems: int = _DEFAULT_CHUNK) -> int:
    """Largest ``cam_cell_id`` on the fine grid where id >= 0 (chunked, no full masked copy)."""
    cid = np.asarray(cam_cell_id, dtype=np.int64, order="C").ravel()
    n = int(cid.size)
    ce = max(10_000, int(chunk_elems))
    mx = -1
    for s in range(0, n, ce):
        e = min(n, s + ce)
        cc = cid[s:e]
        m = cc >= 0
        if np.any(m):
            mx = max(mx, int(cc[m].max()))
    return mx


@dataclass(frozen=True)
class RasterStatsLogger:
    """Small facade carrying sector-specific formatting context."""

    logger: logging.Logger
    tag: str = "debug"

    def _prefix(self) -> str:
        return f"[{self.tag}]"

    def log_raster_stats(
        self,
        name: str,
        arr: np.ndarray,
        *,
        mask: np.ndarray | None = None,
    ) -> None:
        a = np.asarray(arr)
        if mask is not None:
            m = np.asarray(mask, dtype=bool).reshape(a.shape)
            sel = m & np.isfinite(a)
        else:
            sel = np.isfinite(a)
        n = int(a.size)
        n_ok = int(np.count_nonzero(sel))
        if n_ok == 0:
            self.logger.warning(
                "%s %s: shape=%s dtype=%s -- no finite values in selection",
                self._prefix(),
                name,
                a.shape,
                a.dtype,
            )
            return
        v = a[sel]
        self.logger.info(
            "%s %s: shape=%s dtype=%s selection=%d/%d (%.2f%%) min=%.6g max=%.6g mean=%.6g",
            self._prefix(),
            name,
            a.shape,
            a.dtype,
            n_ok,
            n,
            100.0 * n_ok / max(n, 1),
            float(np.min(v)),
            float(np.max(v)),
            float(np.mean(v)),
        )

    def log_population_chain(
        self,
        pop: np.ndarray,
        ref: dict[str, Any],
        p_pop: np.ndarray,
        cam_cell_id: np.ndarray,
    ) -> None:
        area = float(abs(float(ref["transform"][0]) * float(ref["transform"][4])))
        dens = np.where(
            np.isfinite(pop) & (pop >= 0), pop.astype(np.float64) / max(area, 1e-6), np.nan
        )
        cam_ok = cam_cell_id >= 0
        self.log_raster_stats("population (warped raw)", pop, mask=cam_ok)
        self.log_raster_stats(
            "population_density (pop/area_m2)", dens, mask=cam_ok & np.isfinite(dens)
        )
        self.log_raster_stats("P_pop = z(pop_density)", p_pop, mask=cam_ok)
        nz = int(np.count_nonzero((p_pop > 0) & cam_ok))
        tot = int(np.count_nonzero(cam_ok))
        self.logger.info(
            "%s P_pop positive pixels (cam_cell_id>=0): %d / %d (%.2f%%)",
            self._prefix(),
            nz,
            tot,
            100.0 * nz / max(tot, 1),
        )

    def log_cam_and_country(self, cam_cell_id: np.ndarray, country_id: np.ndarray) -> None:
        cid = np.asarray(cam_cell_id).ravel()
        n = cid.size
        n_in = int(np.count_nonzero(cid >= 0))
        n_out = int(np.count_nonzero(cid < 0))
        self.logger.info(
            "%s cam_cell_id: in_domain=%d (%.2f%%) outside=%d (%.2f%%)",
            self._prefix(),
            n_in,
            100.0 * n_in / max(n, 1),
            n_out,
            100.0 * n_out / max(n, 1),
        )
        if n_in:
            self.logger.info(
                "%s cam_cell_id range (valid): [%d, %d] unique~%d",
                self._prefix(),
                int(cid[cid >= 0].min()),
                int(cid[cid >= 0].max()),
                int(np.unique(cid[cid >= 0]).size),
            )
        ctry = np.asarray(country_id).ravel()
        n_ct = int(np.count_nonzero(ctry > 0))
        self.logger.info(
            "%s country_id (NUTS raster): pixels with id>0: %d / %d (%.2f%%)",
            self._prefix(),
            n_ct,
            n,
            100.0 * n_ct / max(n, 1),
        )
