"""Family 1: housing (OSM + LUCAS buildings) vs pasture (CORINE 231/243) with lambda_n."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform

from PROXY.sectors.K_Agriculture.combined_config import compute_lambda_series, load_k_agriculture_rules
from PROXY.sectors.K_Agriculture.signals.osm_farmyard import rasterize_osm_farmyard
from PROXY.sectors.K_Agriculture.source_relevance.lucas_points import get_lucas_ag_points


def build_pasture_layer(corine_arr: np.ndarray, nodata: float | None, rules: dict[str, Any]) -> np.ndarray:
    """Pasture weights on raw CORINE classes 231 / 243."""
    pst = (rules.get("housing_pasture") or {}).get("pasture") or {}
    o231 = float(pst.get("omega_231", 0.9))
    o243 = float(pst.get("omega_243", 0.1))
    x = np.asarray(corine_arr, dtype=np.float64)
    ok = np.isfinite(x)
    if nodata is not None:
        ok = ok & (x != float(nodata))
    out = np.zeros_like(x, dtype=np.float32)
    out[ok & (np.rint(x) == 231)] = np.float32(o231)
    out[ok & (np.rint(x) == 243)] = np.float32(o243)
    return out


def build_housing_layer(
    root: Path,
    ref: dict[str, Any],
    cfg: dict[str, Any],
    *,
    corine_arr: np.ndarray,
    corine_nodata: float | None,
) -> np.ndarray:
    rules = load_k_agriculture_rules(cfg, root)
    hp = rules.get("housing_pasture") or {}
    w_osm = float(hp.get("w_osm", 0.65))
    w_luc = float(hp.get("w_luc", 0.35))
    lu1_req = str(hp.get("lucas_lu1", "U111")).strip().upper()
    lc_build = {str(x).strip().upper() for x in (hp.get("lucas_building_lc1") or ["A11", "A12"])}

    h, w = int(ref["height"]), int(ref["width"])
    osm = rasterize_osm_farmyard(root, ref, cfg)
    osm_n = osm / (float(np.nanmax(osm)) + 1e-12) if np.any(osm > 0) else osm

    luc_r = np.zeros((h, w), dtype=np.float32)
    try:
        pts = get_lucas_ag_points(cfg, root)
    except (FileNotFoundError, ValueError):
        pts = pd.DataFrame()
    if not pts.empty and lc_build:
        lu1 = pts["SURVEY_LU1"].astype(str).str.strip().str.upper()
        lc1 = pts["SURVEY_LC1"].astype(str).str.strip().str.upper()
        m = (lu1 == lu1_req) & lc1.isin(lc_build)
        sub = pts.loc[m]
        if not sub.empty:
            lat = pd.to_numeric(sub["POINT_LAT"], errors="coerce").to_numpy()
            lon = pd.to_numeric(sub["POINT_LONG"], errors="coerce").to_numpy()
            crs_s = ref["crs"].to_string() if hasattr(ref["crs"], "to_string") else str(ref["crs"])
            xs, ys = warp_transform("EPSG:4326", crs_s, lon.tolist(), lat.tolist())
            tf = ref["transform"]
            for x, y in zip(xs, ys):
                try:
                    r, c = rowcol(tf, x, y)
                except Exception:
                    continue
                if 0 <= int(r) < h and 0 <= int(c) < w:
                    luc_r[int(r), int(c)] += 1.0
    if luc_r.max() > 0:
        luc_r = luc_r / (float(luc_r.max()) + 1e-12)

    H = (w_osm * osm_n + w_luc * luc_r).astype(np.float32)
    if float(H.max()) <= 0:
        # fallback: small mass on ag corine if OSM+LUCAS empty
        from PROXY.sectors.K_Agriculture.corine_weight_codes import corine_grid_to_weight_codes

        rint = np.zeros_like(corine_arr, dtype=np.int32)
        ok = np.isfinite(corine_arr)
        if corine_nodata is not None:
            ok = ok & (corine_arr != float(corine_nodata))
        rint[ok] = np.rint(corine_arr[ok]).astype(np.int32)
        wclc = corine_grid_to_weight_codes(rint)
        H = np.where((wclc >= 12) & (wclc <= 22), np.float32(0.1), np.float32(0.0))
    return H


def build_family1(
    root: Path,
    ref: dict[str, Any],
    cfg: dict[str, Any],
    *,
    corine_arr: np.ndarray,
    corine_nodata: float | None,
    nuts_r: np.ndarray,
    nuts_to_idx: dict[str, int],
) -> np.ndarray:
    H = build_housing_layer(root, ref, cfg, corine_arr=corine_arr, corine_nodata=corine_nodata)
    rules = load_k_agriculture_rules(cfg, root)
    P = build_pasture_layer(corine_arr, corine_nodata, rules)
    lam = compute_lambda_series(cfg, root)
    h, w = H.shape
    n_nuts = len(nuts_to_idx) + 1
    idx_to_nuts = {int(v): str(k).strip() for k, v in nuts_to_idx.items()}
    lam_map = np.zeros(n_nuts, dtype=np.float32)
    for idx in range(1, n_nuts):
        nid = idx_to_nuts.get(idx)
        if nid is None:
            lam_map[idx] = 0.6
        else:
            lam_map[idx] = float(lam.get(str(nid).upper(), 0.6))
    ni = np.clip(np.asarray(nuts_r, dtype=np.int32), 0, n_nuts - 1)
    lam_r = lam_map[ni].astype(np.float32)
    return (lam_r * H + (1.0 - lam_r) * P).astype(np.float32)
