"""Sub-proxy: non-road composite (CORINE agri/industrial + OSM industrial + population)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from PROXY.core.osm_corine_proxy import build_p_pop, osm_coverage_fraction, z_score
from Waste.j_waste_weights.io_utils import warp_raster_to_ref

from .constants import INDUSTRIAL_OFFROAD_FAMILIES
from .corine_masks import corine_binary_mask, corine_binary_mask_adapted

logger = logging.getLogger(__name__)


def load_industrial_osm_polygons(osm_gpkg: Path) -> gpd.GeoDataFrame:
    try:
        ar = gpd.read_file(osm_gpkg, layer="osm_offroad_areas")
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    if ar.empty or "offroad_family" not in ar.columns:
        return gpd.GeoDataFrame(geometry=[], crs=ar.crs)
    fam = ar["offroad_family"].astype(str)
    m = fam.isin(INDUSTRIAL_OFFROAD_FAMILIES)
    out = ar.loc[m].copy()
    return out if not out.empty else gpd.GeoDataFrame(geometry=[], crs=ar.crs)


def build_nonroad_proxy(
    *,
    clc_nn: np.ndarray,
    ref: dict[str, Any],
    osm_gpkg: Path,
    proxy_cfg: dict[str, Any],
    pop_raw: np.ndarray,
) -> dict[str, Any]:
    """
    Build non-road composite ``p_nr`` and intermediate z layers.

    ``proxy_cfg`` expects: w_agri, w_ind, w_pop, w_clc_ind, w_osm_ind, corine_agri_codes,
    corine_agri_optional, corine_ind_codes, corine_ind_optional, osm_subdivide.
    """
    subdiv = int(proxy_cfg.get("osm_subdivide", 4))
    w_agri = float(proxy_cfg.get("w_agri", 0.5))
    w_ind = float(proxy_cfg.get("w_ind", 0.35))
    w_pop = float(proxy_cfg.get("w_pop", 0.15))
    w_clc = float(proxy_cfg.get("w_clc_ind", 0.6))
    w_osm = float(proxy_cfg.get("w_osm_ind", 0.4))

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
    z_clc_ind = z_score(ind_mask.astype(np.float64))

    g_ind = load_industrial_osm_polygons(osm_gpkg)
    osm_ind = (
        osm_coverage_fraction(g_ind, ref, subdivide_factor=subdiv)
        if not g_ind.empty
        else np.zeros_like(z_clc_ind, dtype=np.float32)
    )
    z_osm_ind = z_score(osm_ind.astype(np.float64))
    if float(np.nanmax(osm_ind)) <= 1e-12:
        z_ind_comb = z_clc_ind
    else:
        z_ind_comb = (w_clc * z_clc_ind + w_osm * z_osm_ind).astype(np.float32)

    z_pop_term = build_p_pop(pop_raw, ref)

    p_nr = (w_agri * z_agri + w_ind * z_ind_comb + w_pop * z_pop_term).astype(np.float32)
    if float(np.nanmax(p_nr)) <= 1e-16:
        p_nr = z_pop_term.astype(np.float32)

    return {
        "p_nr": p_nr,
        "z_agri": z_agri,
        "z_clc_ind": z_clc_ind,
        "z_osm_ind": z_osm_ind,
        "z_industry_combined": z_ind_comb,
        "z_pop_nonroad": z_pop_term,
    }


def warp_population_to_ref(pop_path: Path, root: Path, ref: dict[str, Any]) -> np.ndarray:
    """Load LandScan (or other) population and warp to ``ref``; zeros if missing."""
    p = pop_path
    if p and not p.is_absolute():
        p = root / p
    if p.is_file():
        return warp_raster_to_ref(p, ref)
    logger.warning("Population raster missing at %s; non-road residential term is zero.", p)
    return np.zeros((int(ref["height"]), int(ref["width"])), dtype=np.float32)
