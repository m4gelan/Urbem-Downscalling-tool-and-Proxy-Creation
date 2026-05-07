"""
CAMS cell → reference-grid **accumulation** (``U = X_w @ Mᵀ`` and per-pollutant share normalisation).

**Role**: isolate the inner numeric loop from NetCDF / window geometry so it can be
unit-tested and profiled independently of I/O.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def accumulate_emissions_and_weights_for_cell(
    *,
    U: np.ndarray,
    pollutant_specs: list[dict[str, Any]],
    ds: xr.Dataset,
    cell_index: int,
    co2_mode: str,
    flat_r: np.ndarray,
    flat_c: np.ndarray,
    acc: np.ndarray,
    weights_acc: np.ndarray | None,
) -> bool:
    """
    Scatter-add one CAMS cell's contribution into ``acc`` (and optionally ``weights_acc``).

    Returns ``True`` if **uniform fallback** was used for at least one pollutant column
    (``sum(U[:,p])`` non-positive or non-finite).
    """
    n_pix = U.shape[0]
    used_fb = False
    for pi, spec in enumerate(pollutant_specs):
        u_col = U[:, pi]
        ssum = float(np.sum(u_col))
        if ssum <= 0.0 or not np.isfinite(ssum):
            w_col = np.full(n_pix, 1.0 / max(n_pix, 1), dtype=np.float64)
            used_fb = True
        else:
            w_col = u_col / ssum
        E = _cams_emission_kg_yr_for_pollutant(ds, cell_index, spec, co2_mode)
        if not np.isfinite(E):
            E = 0.0
        contrib = w_col * E
        np.add.at(acc[pi], (flat_r, flat_c), contrib.astype(np.float64))
        if weights_acc is not None:
            np.add.at(weights_acc[pi], (flat_r, flat_c), w_col)
    return used_fb


def _cams_emission_kg_yr_for_pollutant(
    ds: xr.Dataset,
    i: int,
    spec: dict[str, Any],
    co2_mode: str,
) -> float:
    if spec.get("from_co2_mode"):
        ff = float(np.asarray(ds["co2_ff"].values).ravel()[i])
        bf = float(np.asarray(ds["co2_bf"].values).ravel()[i])
        if co2_mode == "fossil_only":
            return ff
        if co2_mode == "bio_only":
            return bf
        return ff + bf
    var = spec["cams_var"]
    return float(np.asarray(ds[str(var)].values).ravel()[i])
