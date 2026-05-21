from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PROXY_V2.core import log
from PROXY_V2.core.raster_helpers import rasterize_buffered_points
from PROXY_V2.dataset_loaders.load_c21 import load_c21_headcounts
from PROXY_V2.dataset_loaders.load_corine import load_corine_weighted_l3
from PROXY_V2.dataset_loaders.load_lucas import load_lucas_points
from PROXY_V2.dataset_loaders.load_osm import load_osm, rasterize_osm
from PROXY_V2.sector.K_Agriculture.helper import (
    AgReferenceGrid,
    build_ag_reference_grid,
    load_housing_factor_params,
)

_LAMBDA_FALLBACK = 0.6


@dataclass(frozen=True)
class LivestockHousingPastureResult:
    built: np.ndarray
    grazing: np.ndarray
    fused: np.ndarray
    lambda_by_nuts2: dict[str, float]
    ref: AgReferenceGrid


def _norm01(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    mx = float(np.nanmax(a))
    if mx <= 0:
        return a
    return (a / (mx + np.float32(1e-12))).astype(np.float32, copy=False)


def _species_ef_key(species: str, params: dict[str, dict[str, float]]) -> str:
    s = str(species).strip().lower()
    if s not in params:
        raise KeyError(f"species {species!r} missing from housing_factor (have {sorted(params.keys())})")
    return s


def compute_housing_factor(headcounts: dict[str, float], params: dict[str, dict[str, float]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for species, heads in headcounts.items():
        h = float(heads)
        if h <= 0:
            continue
        p = params[_species_ef_key(species, params)]
        n_potential = h * p["liveweight_kg"] * p["N_excretion"]
        numerator += n_potential * p["housing_fraction"]
        denominator += n_potential
    return numerator / denominator if denominator > 0 else _LAMBDA_FALLBACK


def compute_housing_factor_by_nuts2(
    headcounts_df: Any,
    params: dict[str, dict[str, float]],
    *,
    country_nuts_ids: set[str] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for nuts_id, row in headcounts_df.iterrows():
        nid = str(nuts_id).strip().upper()
        counts = {str(c): float(row[c]) for c in headcounts_df.columns}
        lam = compute_housing_factor(counts, params)
        out[nid] = lam
        if country_nuts_ids is None or nid in country_nuts_ids:
            log.debug(f"lambda_H {nid}: {lam:.4f}")
    if country_nuts_ids:
        sub = {k: v for k, v in out.items() if k in country_nuts_ids}
        log.info(
            f"lambda_H ({len(sub)} NUTS2 in country): "
            + " ".join(f"{k}={v:.4f}" for k, v in sorted(sub.items()))
        )
    elif out:
        vals = list(out.values())
        log.info(
            f"lambda_H: {len(out)} NUTS2 regions "
            f"min={min(vals):.4f} max={max(vals):.4f} mean={sum(vals)/len(vals):.4f}"
        )
    return out


def _pasture_l3_weights(corine_cfg: dict[str, Any], pasture_weights: dict[str, Any]) -> dict[int, float]:
    codes = [int(x) for x in (corine_cfg.get("pasture_l3_codes") or [])]
    if not codes:
        raise ValueError("corine.pasture_l3_codes required")
    out: dict[int, float] = {}
    for l3 in codes:
        wkey = f"w_clc{l3}"
        if wkey not in pasture_weights:
            raise ValueError(f"weights.livestock_housing_pasture.pasture.{wkey} required")
        out[l3] = float(pasture_weights[wkey])
    return out


def _lambda_raster(
    nuts_r: np.ndarray,
    nuts_id_to_idx: dict[str, int],
    lambda_by_nuts2: dict[str, float],
) -> np.ndarray:
    idx_to_nuts = {int(v): str(k).strip().upper() for k, v in nuts_id_to_idx.items()}
    n_nuts = len(nuts_id_to_idx) + 1
    lam_map = np.full(n_nuts, _LAMBDA_FALLBACK, dtype=np.float32)
    for idx, nid in idx_to_nuts.items():
        lam_map[idx] = np.float32(lambda_by_nuts2.get(nid, _LAMBDA_FALLBACK))
    ni = np.clip(np.asarray(nuts_r, dtype=np.int32), 0, n_nuts - 1)
    return lam_map[ni]


def _build_housing_osm(
    repo_root: Path,
    osm_path: Path,
    osm_cfg: dict[str, Any],
    ref: AgReferenceGrid,
) -> np.ndarray:
    gdf = load_osm(repo_root / str(osm_path).replace("\\", "/"), ref.cams_cells, osm_cfg)
    rz = osm_cfg.get("rasterize") or {}
    raw = rasterize_osm(
        gdf, ref.height, ref.width, ref.transform, ref.crs, rz, ref.cams_cells,
    )
    log.info(f"housing OSM raster: sum={float(raw.sum()):.6g} max={float(raw.max()):.6g}")
    return _norm01(raw)


def _build_housing_lucas(
    repo_root: Path,
    lucas_path: Path,
    country_profile: dict[str, str],
    lucas_root: dict[str, Any],
    lucas_signal: dict[str, Any],
    ref: AgReferenceGrid,
    osm_cfg: dict[str, Any],
) -> np.ndarray:
    pts = load_lucas_points(
        repo_root / str(lucas_path).replace("\\", "/"),
        country_profile,
        lucas_signal,
        lucas_root_cfg=lucas_root,
    )
    if pts.empty:
        log.info("housing LUCAS: no points — zero layer")
        return np.zeros((ref.height, ref.width), dtype=np.float32)
    buf = float(lucas_signal.get("buffer_m"))
    raw = rasterize_buffered_points(
        pts["lon"].to_numpy(),
        pts["lat"].to_numpy(),
        buffer_m=buf,
        metric_crs=str(osm_cfg["metric_crs"]),
        height=ref.height,
        width=ref.width,
        transform=ref.transform,
        raster_crs=ref.crs,
        burn_value=float((osm_cfg.get("rasterize") or {}).get("burn_value", 1.0)),
        fill=0.0,
        dtype=np.float32,
        all_touched=bool((osm_cfg.get("rasterize") or {}).get("all_touched", True)),
    )
    log.info(f"housing LUCAS raster (buffer={buf}m): sum={float(raw.sum()):.6g}")
    return _norm01(raw)


def _combine_housing(H_osm: np.ndarray, H_lucas: np.ndarray, w_osm: float, w_lucas: float) -> np.ndarray:
    H = (np.float32(w_osm) * H_osm + np.float32(w_lucas) * H_lucas).astype(np.float32)
    log.info(f"housing combined: w_osm={w_osm} w_lucas={w_lucas} max={float(H.max()):.6g}")
    return H


def _build_pasture_corine(
    repo_root: Path,
    corine_path: Path,
    corine_band: int,
    l3_weights: dict[int, float],
    ref: AgReferenceGrid,
) -> np.ndarray:
    acc = load_corine_weighted_l3(
        repo_root / str(corine_path).replace("\\", "/"),
        l3_weights,
        corine_band,
        ref.cams_cells,
        ref_height=ref.height,
        ref_width=ref.width,
        ref_transform=ref.transform,
        ref_crs=ref.crs,
    )
    log.info(f"pasture CORINE blend: sum={float(acc.sum()):.6g} max={float(acc.max()):.6g}")
    return _norm01(acc)


def _build_pasture_lucas(
    repo_root: Path,
    lucas_path: Path,
    country_profile: dict[str, str],
    lucas_root: dict[str, Any],
    lucas_signal: dict[str, Any],
    ref: AgReferenceGrid,
    osm_cfg: dict[str, Any],
) -> np.ndarray:
    pts = load_lucas_points(
        repo_root / str(lucas_path).replace("\\", "/"),
        country_profile,
        lucas_signal,
        lucas_root_cfg=lucas_root,
    )
    if pts.empty:
        log.info("pasture LUCAS: no points — zero layer")
        return np.zeros((ref.height, ref.width), dtype=np.float32)
    buf = float(lucas_signal.get("buffer_m"))
    raw = rasterize_buffered_points(
        pts["lon"].to_numpy(),
        pts["lat"].to_numpy(),
        buffer_m=buf,
        metric_crs=str(osm_cfg["metric_crs"]),
        height=ref.height,
        width=ref.width,
        transform=ref.transform,
        raster_crs=ref.crs,
        burn_value=1.0,
        fill=0.0,
        dtype=np.float32,
        all_touched=True,
    )
    log.info(f"pasture LUCAS raster (buffer={buf}m): sum={float(raw.sum()):.6g}")
    return _norm01(raw)


def _combine_pasture(P_corine: np.ndarray, P_lucas: np.ndarray, w_corine: float, w_lucas: float) -> np.ndarray:
    if w_lucas <= 0:
        P = P_corine
    else:
        P = (np.float32(w_corine) * P_corine + np.float32(w_lucas) * P_lucas).astype(np.float32)
    log.info(f"grazing/pasture combined: max={float(P.max()):.6g} sum={float(P.sum()):.6g}")
    return P


def build_livestock_housing_pasture(
    repo_root: Path,
    cfg: dict[str, Any],
    country_profile: dict[str, str],
    *,
    sector_config_path: Path,
    cams_cells: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
    corine_filepath: str | Path,
    osm_filepath: str | Path,
    nuts_filepath: str | Path,
    farmstock_filepath: str | Path,
    lucas_filepath: str | Path,
) -> LivestockHousingPastureResult:
    log.info("--- K_Agriculture signal: livestock housing + pasture (3.B) ---")

    corine_cfg = cfg.get("corine") or {}
    corine_band = int(corine_cfg.get("band", 1))
    osm_cfg = cfg.get("osm") or {}
    lucas_cfg = cfg.get("LUCAS") or {}
    farmstock_cfg = cfg.get("Farmstock") or {}
    weights = (cfg.get("weights") or {}).get("livestock_housing_pasture") or {}
    built_w = weights.get("built") or {}
    pasture_w = weights.get("pasture") or {}

    ref = build_ag_reference_grid(
        repo_root,
        corine_path=corine_filepath,
        corine_band=corine_band,
        corine_l3_anchor=[121],
        nuts_path=nuts_filepath,
        country_profile=country_profile,
        cams_cells=cams_cells,
        cams_grid=cams_grid,
    )
    country_nuts = {str(k).strip().upper() for k in ref.nuts_id_to_idx}

    ef_path = sector_config_path.parent / "emission_factors.yaml"
    params = load_housing_factor_params(ef_path)
    headcounts = load_c21_headcounts(
        repo_root / str(farmstock_filepath).replace("\\", "/"),
        farmstock_cfg,
    )
    headcounts = headcounts.loc[
        headcounts.index.astype(str).str.strip().str.upper().isin(country_nuts)
    ]
    log.info(f"C21 headcounts for country: {len(headcounts)} NUTS2 rows")
    lambda_by_nuts2 = compute_housing_factor_by_nuts2(
        headcounts, params, country_nuts_ids=country_nuts,
    )
    l3_weights = _pasture_l3_weights(corine_cfg, pasture_w)

    H_osm = _build_housing_osm(repo_root, osm_filepath, osm_cfg, ref)
    H_lucas = _build_housing_lucas(
        repo_root,
        lucas_filepath,
        country_profile,
        lucas_cfg,
        lucas_cfg.get("livestock_housing") or {},
        ref,
        osm_cfg,
    )
    H = _combine_housing(
        H_osm,
        H_lucas,
        float(built_w.get("w_osm")),
        float(built_w.get("w_lucas")),
    )

    P_corine = _build_pasture_corine(
        repo_root, corine_filepath, corine_band, l3_weights, ref,
    )
    P_lucas = _build_pasture_lucas(
        repo_root,
        lucas_filepath,
        country_profile,
        lucas_cfg,
        lucas_cfg.get("pasture") or {},
        ref,
        osm_cfg,
    )
    w_lucas_p = float(pasture_w.get("w_lucas", 0.0))
    grazing = _combine_pasture(P_corine, P_lucas, 1.0 - w_lucas_p, w_lucas_p)

    lam_r = _lambda_raster(ref.nuts_r, ref.nuts_id_to_idx, lambda_by_nuts2)
    fused = (lam_r * H + (np.float32(1.0) - lam_r) * grazing).astype(np.float32)
    log.info(
        f"livestock housing/pasture fused: max={float(fused.max()):.6g} "
        f"built_max={float(H.max()):.6g} grazing_max={float(grazing.max()):.6g}"
    )

    return LivestockHousingPastureResult(
        built=H,
        grazing=grazing,
        fused=fused,
        lambda_by_nuts2=lambda_by_nuts2,
        ref=ref,
    )
