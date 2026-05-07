"""Non-road (1A3eii): CORINE class weights (1A3eii residual), no OSM construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from PROXY.core.corine.raster import corine_binary_mask, corine_binary_mask_adapted
from PROXY.core.osm_corine_proxy import clc_weighted_class_score, z_score


def build_nonroad_corine_proxy(
    *,
    clc_nn: np.ndarray,
    proxy_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Build ``p_nr`` from CORINE classes and configured weights, then :func:`z_score`.

    If ``nonroad_corine_weights`` is set, it is a ``{clc_code: weight}`` mapping
    (sum need not be 1; it is renormalised in :func:`clc_weighted_class_score`).

    Legacy: ``w_agri`` / ``w_ind`` with agri / ind code lists (unchanged if no new block).
    """
    nw = proxy_cfg.get("nonroad_corine_weights")
    if nw:
        weights = {}
        for k, v in dict(nw).items():
            try:
                weights[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
        raw = clc_weighted_class_score(clc_nn, weights).astype(np.float64)
        p_nr = z_score(raw)
        return {
            "p_nr": p_nr,
            "nonroad_corine_raw": raw.astype(np.float32),
        }

    w_agri = float(proxy_cfg.get("w_agri", 0.5))
    w_ind = float(proxy_cfg.get("w_ind", 0.35))

    agri_codes = [int(x) for x in (proxy_cfg.get("corine_agri_codes") or [])]
    agri_codes += [int(x) for x in (proxy_cfg.get("corine_agri_optional") or [])]
    ind_codes = [int(x) for x in (proxy_cfg.get("corine_ind_codes") or [])]
    ind_codes += [int(x) for x in (proxy_cfg.get("corine_ind_optional") or [])]

    from PROXY.core.corine.raster import corine_binary_mask, corine_binary_mask_adapted

    agri_mask = corine_binary_mask(clc_nn, agri_codes)
    if float(np.max(agri_mask)) <= 0 and agri_codes:
        agri_mask2, _ = corine_binary_mask_adapted(clc_nn, agri_codes)
        agri_mask = np.maximum(agri_mask, agri_mask2)

    ind_mask = (
        corine_binary_mask_adapted(clc_nn, ind_codes)[0]
        if ind_codes
        else np.zeros_like(clc_nn, dtype=np.float32)
    )

    z_agri = z_score(agri_mask.astype(np.float64))
    z_ind = z_score(ind_mask.astype(np.float64))
    p_nr = (w_agri * z_agri + w_ind * z_ind).astype(np.float32)

    return {
        "p_nr": p_nr,
        "z_agri": z_agri,
        "z_industry_corine": z_ind,
    }
