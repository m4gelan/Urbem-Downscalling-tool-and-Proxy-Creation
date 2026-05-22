from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.enums import MergeAlg
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from proxy.core import log
from proxy.dataset_loaders.load_cams_cells_mask import (
    _union_bounds_wgs84,
    pixels_inside_cams_cells,
)

_VIIRS_SHP_GLOB = "fire_archive_SV-C2_*.shp"
_VIIRS_COLS = ("LATITUDE", "LONGITUDE", "ACQ_DATE", "FRP", "TYPE", "CONFIDENCE")


def viirs_active_fire_shp(folder: Path) -> Path:
    hits = sorted(Path(folder).glob(_VIIRS_SHP_GLOB))
    if not hits:
        raise FileNotFoundError(f"No {_VIIRS_SHP_GLOB} in {folder}")
    return hits[0]


def load_viirs_active_fires(
    viirs_dir: Path,
    cams_cells: dict[int, dict[str, Any]],
) -> pd.DataFrame:
    """FIRMS VIIRS S-NPP C2 points inside the CAMS WGS84 bounding box (no geometry column)."""
    shp = viirs_active_fire_shp(viirs_dir)
    w, s, e, n = _union_bounds_wgs84(cams_cells)
    bbox = (w, s, e, n)
    try:
        g = gpd.read_file(shp, bbox=bbox, ignore_geometry=True)
    except TypeError:
        g = gpd.read_file(shp, bbox=bbox)
        g = g.drop(columns="geometry", errors="ignore")

    missing = [c for c in _VIIRS_COLS if c not in g.columns]
    if missing:
        raise ValueError(f"VIIRS shapefile missing columns {missing}; have {list(g.columns)[:20]}")

    g = g.loc[:, list(_VIIRS_COLS)].copy()
    g = g[
        (g["LONGITUDE"] >= w)
        & (g["LONGITUDE"] <= e)
        & (g["LATITUDE"] >= s)
        & (g["LATITUDE"] <= n)
    ]
    g["FRP"] = pd.to_numeric(g["FRP"], errors="coerce").astype(np.float32)
    g["LATITUDE"] = g["LATITUDE"].astype(np.float32)
    g["LONGITUDE"] = g["LONGITUDE"].astype(np.float32)
    g["TYPE"] = g["TYPE"].astype(np.int16)
    log.info(f"VIIRS active fire: {len(g)} points in CAMS bbox ({shp.name})")
    return g


def filter_viirs_active_fires(df: pd.DataFrame, viirs_cfg: dict[str, Any]) -> pd.DataFrame:
    """Apply sector ``VIIRS`` block: type, confidence, frp_threshold."""
    types = {int(x) for x in viirs_cfg["type"]}
    conf = {str(c).strip().lower() for c in viirs_cfg["confidence"]}
    frp_min = np.float32(viirs_cfg["frp_threshold"])

    t = df["TYPE"].astype(int)
    c = df["CONFIDENCE"].astype(str).str.strip().str.lower()
    frp = df["FRP"]
    keep = t.isin(types) & c.isin(conf) & (frp >= frp_min)
    out = df.loc[keep].copy()
    log.info(
        f"VIIRS filter: {len(out)}/{len(df)} kept "
        f"(type in {sorted(types)}, conf in {sorted(conf)}, FRP>={float(frp_min)})"
    )
    return out


def _cams_domain_polygon_wgs84(cams_cells: dict[int, dict[str, Any]]):
    polys = []
    for c in cams_cells.values():
        b = c["cell_bounds_wgs84"]
        polys.append(box(b["west"], b["south"], b["east"], b["north"]))
    return unary_union(polys)


def _dedupe_max_rhi(df: pd.DataFrame, lat_c: str, lon_c: str, rhi_c: str) -> pd.DataFrame:
    df = df.copy()
    df["_la"] = df[lat_c].round(5)
    df["_lo"] = df[lon_c].round(5)
    df = df.sort_values(rhi_c, ascending=False).drop_duplicates(subset=["_la", "_lo"], keep="first")
    return df.drop(columns=["_la", "_lo"])


def load_vnf_nightfire_buffer_mask(
    csv_path: Path,
    h: int,
    w: int,
    transform: Any,
    crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
) -> np.ndarray:
    """VNF CSV: flare points buffered by ``cfg['buffer_m']``, clipped to CAMS cell union (WGS84), 0/1 grid."""
    lat_c = str(cfg["lat_column"])
    lon_c = str(cfg["lon_column"])
    rhi_c = str(cfg["rhi_column"])
    buf_m = float(cfg["buffer_m"])
    rhi_bad = float(cfg["rhi_sentinel_max"])

    df = pd.read_csv(csv_path, low_memory=False)
    for c in (lat_c, lon_c, rhi_c):
        if c not in df.columns:
            raise ValueError(f"VNF CSV missing column {c!r}; have {list(df.columns)[:30]}")
    df = df.replace([rhi_bad, 999999], np.nan)
    df = df[np.isfinite(df[lat_c]) & np.isfinite(df[lon_c])]
    df = df[(df[rhi_c] < 9.0e5) & (df[rhi_c] > 0) & np.isfinite(df[rhi_c])]
    if df.empty:
        log.info("VNF: no valid flare rows after RHI/lat-lon filter")
        return np.zeros((h, w), dtype=np.float32)

    domain = _cams_domain_polygon_wgs84(cams_cells)
    g = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_c], df[lat_c], crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    g = g[g.intersects(domain)]
    if g.empty:
        log.info("VNF: no rows inside CAMS WGS84 footprint")
        return np.zeros((h, w), dtype=np.float32)
    g = _dedupe_max_rhi(g, lat_c, lon_c, rhi_c)
    crs_tgt = str(crs)
    g = g.to_crs(crs_tgt)
    g["geometry"] = g.geometry.buffer(buf_m)

    acc = np.zeros((h, w), dtype=np.float32)
    shapes = ((mapping(geom), 1.0) for geom in g.geometry if geom is not None and not geom.is_empty)
    features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=0.0,
        out=acc,
        dtype=np.float32,
        all_touched=True,
        merge_alg=MergeAlg.replace,
    )
    inside = pixels_inside_cams_cells(h, w, transform, crs, cams_cells)
    acc = np.where(inside, acc, 0.0).astype(np.float32)
    return (acc > 0.5).astype(np.float32)
