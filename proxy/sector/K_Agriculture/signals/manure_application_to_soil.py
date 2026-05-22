from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from proxy.core import log
from proxy.core.z_score import z_score_inside
from proxy.dataset_loaders.load_c21 import load_c21_headcounts
from proxy.dataset_loaders.load_corine import load_corine_crop_groups
from proxy.dataset_loaders.load_lucas import rasterize_lucas_crop_groups
from proxy.writers.debug_dump import ManureNutsDebug
from proxy.sector.K_Agriculture.helper import (
    AgReferenceGrid,
    build_manure_nuts_pools,
    crop_groups_from_lucas_cfg,
    load_housing_factor_params,
    load_manure_application_params,
    pixel_area_ha,
)


@dataclass(frozen=True)
class ManureApplicationResult:
    kg_n_per_pixel_yr: np.ndarray
    z_scored: np.ndarray
    ref: AgReferenceGrid
    manure_debug: tuple[ManureNutsDebug, ...] | None = None


def _merge_crop_group_raster(
    repo_root: Path,
    corine_path: Path,
    corine_band: int,
    corine_cfg: dict[str, Any],
    cams_cells: dict[int, dict[str, Any]],
    lucas_path: Path,
    country_profile: dict[str, str],
    lucas_cfg: dict[str, Any],
    manure_blocks: dict[str, Any],
    ref: AgReferenceGrid,
    osm_cfg: dict[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    crop_groups, group_to_idx = crop_groups_from_lucas_cfg(lucas_cfg)
    priority = crop_groups

    gid = load_corine_crop_groups(
        repo_root / str(corine_path).replace("\\", "/"),
        corine_band,
        corine_cfg,
        crop_groups,
        cams_cells,
        priority=priority,
        ref_height=ref.height,
        ref_width=ref.width,
        ref_transform=ref.transform,
        ref_crs=ref.crs,
    )
    rz = osm_cfg.get("rasterize") or {}
    lucas_masks = rasterize_lucas_crop_groups(
        repo_root / str(lucas_path).replace("\\", "/"),
        country_profile,
        lucas_cfg,
        manure_blocks,
        crop_groups,
        height=ref.height,
        width=ref.width,
        transform=ref.transform,
        raster_crs=ref.crs,
        metric_crs=str(osm_cfg["metric_crs"]),
        burn_value=float(rz.get("burn_value", 1.0)),
        all_touched=bool(rz.get("all_touched", True)),
    )
    for gname in reversed(priority):
        m = lucas_masks[gname] > 0
        if m.any():
            gid[m] = group_to_idx[gname]
    log.info(
        "crop groups (CLC+LUCAS): "
        + " ".join(f"{g}={int((gid == group_to_idx[g]).sum())}" for g in crop_groups)
    )
    return gid, crop_groups, group_to_idx


def _zonal_areas_ha(
    gid: np.ndarray,
    nuts_r: np.ndarray,
    area_ha: np.ndarray,
    n_nuts: int,
    crop_groups: list[str],
    group_to_idx: dict[str, int],
) -> dict[str, np.ndarray]:
    flat_gid = gid.ravel()
    flat_nuts = nuts_r.ravel().astype(np.int32)
    flat_area = area_ha.ravel()
    out = {g: np.zeros(n_nuts + 1, dtype=np.float64) for g in crop_groups}
    for gname, gidx in group_to_idx.items():
        m = flat_gid == gidx
        if not m.any():
            continue
        np.add.at(out[gname], flat_nuts[m], flat_area[m])
    return out


def _build_S_kg_n(
    gid: np.ndarray,
    nuts_r: np.ndarray,
    area_ha: np.ndarray,
    pools: dict[str, dict[str, float]],
    nuts_id_to_idx: dict[str, int],
    rate_ratios: dict[str, float],
    crop_groups: list[str],
    group_to_idx: dict[str, int],
    manure_debug: list[ManureNutsDebug] | None = None,
) -> np.ndarray:
    n_nuts = len(nuts_id_to_idx) + 1
    idx_to_nuts = {int(v): str(k).strip().upper() for k, v in nuts_id_to_idx.items()}
    areas = _zonal_areas_ha(gid, nuts_r, area_ha, n_nuts, crop_groups, group_to_idx)
    n_g = len(crop_groups)

    R = np.zeros((n_nuts, n_g), dtype=np.float64)
    for idx in range(1, n_nuts):
        nid = idx_to_nuts.get(idx)
        if not nid or nid not in pools:
            continue
        p = pools[nid]
        g_fod, g_hi, g_med, g_lo = crop_groups
        Af = areas[g_fod][idx]
        Ah = areas[g_hi][idx]
        Am = areas[g_med][idx]
        Al = areas[g_lo][idx]
        R[idx, 0] = p["N_fodder"] / Af if Af > 0 else 0.0
        denom = Ah + rate_ratios[g_med] * Am + rate_ratios[g_lo] * Al
        r_i = p["N_nonfodder"] / denom if denom > 0 else 0.0
        R[idx, 1] = r_i
        R[idx, 2] = r_i * rate_ratios[g_med]
        R[idx, 3] = r_i * rate_ratios[g_lo]
        if manure_debug is not None:
            manure_debug.append(
                ManureNutsDebug(
                    nuts2=nid,
                    n_fodder=float(p["N_fodder"]),
                    n_nonfodder=float(p["N_nonfodder"]),
                    n_total=float(p["N_total"]),
                    r_fodder=float(R[idx, 0]),
                    r_high=float(R[idx, 1]),
                    r_med=float(R[idx, 2]),
                    r_low=float(R[idx, 3]),
                    ha_fodder=float(Af),
                    ha_high=float(Ah),
                    ha_med=float(Am),
                    ha_low=float(Al),
                )
            )
        else:
            log.debug(
                f"manure rates {nid}: R_fodder={R[idx,0]:.4g} R_high={R[idx,1]:.4g} "
                f"ha F={Af:.2e} I={Ah:.2e} II={Am:.2e} III={Al:.2e}"
            )

    ni = np.clip(nuts_r.astype(np.int32), 0, n_nuts - 1)
    S = np.zeros(gid.shape, dtype=np.float64)
    crop = gid > 0
    for gi, gname in enumerate(crop_groups):
        gidx = group_to_idx[gname]
        m = crop & (gid == gidx)
        if m.any():
            S[m] = area_ha[m] * R[ni[m], gi]

    for idx in range(1, n_nuts):
        nid = idx_to_nuts.get(idx)
        if not nid or nid not in pools:
            continue
        m = nuts_r == idx
        s = float(S[m].sum())
        target = float(pools[nid]["N_total"])
        if s > 0 and target > 0:
            S[m] *= target / s
    return S.astype(np.float32)


def build_manure_application_to_soils(
    repo_root: Path,
    cfg: dict[str, Any],
    country_profile: dict[str, str],
    *,
    sector_config_path: Path,
    ref: AgReferenceGrid,
    cams_cells: dict[int, dict[str, Any]],
    corine_filepath: str | Path,
    lucas_filepath: str | Path,
    farmstock_filepath: str | Path,
) -> ManureApplicationResult:
    log.info("--- K_Agriculture signal: manure application to soils (3.Da2a) ---")

    corine_cfg = cfg.get("corine") or {}
    corine_band = int(corine_cfg["band"])
    lucas_cfg = cfg.get("LUCAS") or {}
    farmstock_cfg = cfg.get("Farmstock") or {}
    osm_cfg = cfg.get("osm") or {}
    manure_blocks = lucas_cfg.get("manure_application_to_soils") or {}
    crop_groups, _ = crop_groups_from_lucas_cfg(lucas_cfg)

    ef_path = sector_config_path.parent / "emission_factors.yaml"
    housing_params = load_housing_factor_params(ef_path)
    manure_ef = load_manure_application_params(ef_path)
    L = float(manure_ef["storage_n_loss_fraction"])
    rate_ratios = manure_ef["nonfodder_rate_ratios"]
    for g in crop_groups[1:]:
        if g not in rate_ratios:
            raise ValueError(f"nonfodder_rate_ratios missing {g!r}")

    country_nuts = {str(k).strip().upper() for k in ref.nuts_id_to_idx}
    headcounts = load_c21_headcounts(
        repo_root / str(farmstock_filepath).replace("\\", "/"),
        farmstock_cfg,
    )
    headcounts = headcounts.loc[
        headcounts.index.astype(str).str.strip().str.upper().isin(country_nuts)
    ]
    pools = build_manure_nuts_pools(
        headcounts,
        housing_params,
        L,
        country_profile,
        country_nuts_ids=country_nuts,
    )

    gid, crop_groups, group_to_idx = _merge_crop_group_raster(
        repo_root,
        corine_filepath,
        corine_band,
        corine_cfg,
        cams_cells,
        lucas_filepath,
        country_profile,
        lucas_cfg,
        manure_blocks,
        ref,
        osm_cfg,
    )
    area_ha = pixel_area_ha(ref.transform, ref.height, ref.width)
    dbg_on = log.debug_enabled()
    m_dbg: list[ManureNutsDebug] | None = [] if dbg_on else None
    S = _build_S_kg_n(
        gid,
        ref.nuts_r,
        area_ha,
        pools,
        ref.nuts_id_to_idx,
        rate_ratios,
        crop_groups,
        group_to_idx,
        manure_debug=m_dbg,
    )

    inside = (ref.nuts_r > 0) & (gid > 0)
    z = z_score_inside(S, inside, upper_quantile=0.99, rescale_to_01=True)
    log.info(
        f"manure application: S sum={float(S.sum()):.6g} max={float(S.max()):.6g} "
        f"z max={float(z.max()):.6g} crop px={int(inside.sum())}"
    )
    return ManureApplicationResult(
        kg_n_per_pixel_yr=S,
        z_scored=z,
        ref=ref,
        manure_debug=tuple(m_dbg) if m_dbg is not None else None,
    )
