from __future__ import annotations

from typing import Any

import numpy as np

from proxy.core import log
from proxy.core.area_weights import normalize_W_per_cams_cell


def build_x_tot_by_class(
    pi: dict[str, dict[str, float]],
    aadt_rasters: dict[str, np.ndarray],
    road_types: list[str],
    classes: list[str],
) -> dict[str, np.ndarray]:
    """X_c,tot(j) = sum_r Pi_r,c * L_r(j)."""
    ref = next(iter(aadt_rasters.values()))
    out: dict[str, np.ndarray] = {}
    for c in classes:
        acc = np.zeros_like(ref, dtype=np.float32)
        for r in road_types:
            acc += float(pi[r][c]) * aadt_rasters[r]
        out[c] = acc
    return out


def _x_band(ci: int, fi: int, n_fuels: int) -> int:
    return ci * n_fuels + fi


def build_category_weight_stack(
    *,
    x: np.ndarray | None,
    x_tot: dict[str, np.ndarray],
    m_exh: np.ndarray,
    m_non: np.ndarray,
    classes: list[str],
    fuels: list[str],
    pollutants: list[str],
    category: str,
    category_fuels: list[str] | None,
    cell_id: np.ndarray,
    cams_cells: dict[int, dict[str, Any]],
) -> np.ndarray:
    """Return (n_poll, H, W) normalized weights for one CAMS F subcategory."""
    if category == "F4":
        ref = next(iter(x_tot.values()))
        h, w = ref.shape
    else:
        if x is None:
            raise ValueError("F_Roads exhaust categories require X stack")
        h, w, _ = x.shape
    n_p = len(pollutants)
    fuel_ix = {f: i for i, f in enumerate(fuels)}
    stack = np.zeros((n_p, h, w), dtype=np.float32)
    u = np.zeros((h, w), dtype=np.float32)

    for pj, pol in enumerate(pollutants):
        u.fill(0.0)
        if category == "F4":
            for ci, c in enumerate(classes):
                m = float(m_non[ci, pj])
                if m > 0:
                    u += x_tot[c] * m
        else:
            for f in category_fuels or []:
                fi = fuel_ix[f]
                for ci in range(len(classes)):
                    m = float(m_exh[ci, fi, pj])
                    if m <= 0:
                        continue
                    band = _x_band(ci, fi, len(fuels))
                    u += x[:, :, band] * m
        stack[pj] = normalize_W_per_cams_cell(u, cell_id, cams_cells)

    log.info(f"F_Roads W_{category} stack shape={stack.shape}")
    return stack
