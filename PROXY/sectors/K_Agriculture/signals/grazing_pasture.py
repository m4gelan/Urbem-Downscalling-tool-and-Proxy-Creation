"""Family 3: grazed pasture with LUCAS-confirmed grazing amplification per CORINE pasture polygon."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rasterio import features
from rasterio.warp import transform as warp_transform
from shapely.geometry import Point, shape

from PROXY.sectors.K_Agriculture.combined_config import load_k_agriculture_rules


def _rules(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    return load_k_agriculture_rules(cfg, root)


def polygon_grazing_weight(
    count: int,
    max_count: int,
    *,
    kappa: float,
    epsilon: float,
    w_max: float,
) -> float:
    """Per-polygon multiplier w_m = min(w_max, 1 + kappa * n_tilde) with n_tilde = count / (max_count + epsilon)."""
    nmax = int(max_count)
    nm = int(count)
    nt = nm / (nmax + float(epsilon)) if nmax > 0 else 0.0
    return float(min(float(w_max), 1.0 + float(kappa) * nt))


def build_family3(
    root: Path,
    ref: dict[str, Any],
    cfg: dict[str, Any],
    *,
    corine_arr: np.ndarray,
    corine_nodata: float | None,
) -> np.ndarray:
    h, w = int(ref["height"]), int(ref["width"])
    tf = ref["transform"]
    crs_s = ref["crs"].to_string() if hasattr(ref["crs"], "to_string") else str(ref["crs"])

    rules = _rules(cfg, root)
    gz = (rules.get("grazing") or {})
    kappa = float(gz.get("kappa", 0.5))
    eps = float(gz.get("epsilon", 1e-6))
    w_max = float(gz.get("w_max", 4.0))

    x = np.asarray(corine_arr, dtype=np.float64)
    ok = np.isfinite(x)
    if corine_nodata is not None:
        ok = ok & (x != float(corine_nodata))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[ok & (np.rint(x) == 231)] = 1
    mask[ok & (np.rint(x) == 243)] = 1

    weights = np.ones((h, w), dtype=np.float64)
    if mask.any():
        polys: list[tuple[Any, int]] = []
        for geom, val in features.shapes(mask, mask=mask, transform=tf, connectivity=8):
            if int(val) == 0:
                continue
            polys.append((shape(geom), 0))

        try:
            pts = get_lucas_ag_points(cfg, root)
        except (FileNotFoundError, ValueError):
            pts = pd.DataFrame()

        counts = [0] * len(polys)
        if not pts.empty and polys:
            graze = pd.to_numeric(pts["SURVEY_GRAZING"], errors="coerce")
            mpt = graze == 1.0
            sub = pts.loc[mpt]
            if not sub.empty:
                lat = pd.to_numeric(sub["POINT_LAT"], errors="coerce").to_numpy()
                lon = pd.to_numeric(sub["POINT_LONG"], errors="coerce").to_numpy()
                xs, ys = warp_transform("EPSG:4326", crs_s, lon.tolist(), lat.tolist())

                for x0, y0 in zip(xs, ys):
                    pt = Point(float(x0), float(y0))
                    for i, (poly, _) in enumerate(polys):
                        if poly.contains(pt) or poly.touches(pt):
                            counts[i] += 1
        nmax = max(counts) if counts else 0
        for i, (poly, _) in enumerate(polys):
            wm = polygon_grazing_weight(counts[i], nmax, kappa=kappa, epsilon=eps, w_max=w_max)
            gmask = features.rasterize([(poly, float(wm))], out_shape=(h, w), transform=tf, fill=1.0, dtype=np.float64)
            weights = np.maximum(weights, gmask)

    out = mask.astype(np.float64) * weights
    mx = float(out.max()) if np.any(out > 0) else 1.0
    return (out / mx).astype(np.float32)
