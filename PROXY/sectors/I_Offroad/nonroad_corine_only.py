"""Non-road (1A3eii): CORINE agricultural + industrial classes only (no population / OSM industrial)."""

from __future__ import annotations

from typing import Any

import numpy as np

from PROXY.core.osm_corine_proxy import z_score

from PROXY.core.corine.raster import corine_binary_mask, corine_binary_mask_adapted


def build_nonroad_corine_proxy(
    *,
    clc_nn: np.ndarray,
    proxy_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Build ``p_nr`` from CORINE agri + industrial masks only.

    ``proxy_cfg`` expects: w_agri, w_ind, corine_agri_codes, corine_agri_optional,
    corine_ind_codes, corine_ind_optional.
    """
    w_agri = float(proxy_cfg.get("w_agri", 0.5))
    w_ind = float(proxy_cfg.get("w_ind", 0.35))

    agri_codes = [int(x) for x in (proxy_cfg.get("corine_agri_codes") or [])]
    agri_codes += [int(x) for x in (proxy_cfg.get("corine_agri_optional") or [])]
    ind_codes = [int(x) for x in (proxy_cfg.get("corine_ind_codes") or [])]
    ind_codes += [int(x) for x in (proxy_cfg.get("corine_ind_optional") or [])]

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
