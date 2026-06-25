from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.enums import MergeAlg
from shapely.geometry import Point, mapping

from proxy.core import log
from proxy.dataset_loaders.load_cams_cells_mask import pixels_inside_cams_cells


def _norm_col(c: Any) -> str:
    return str(c).replace("\n", " ").strip().lower()


def _country_rows_eq(series: pd.Series, country_full_name: str) -> pd.Series:
    """GEM ``Country / Area`` cells use the same country wording as ``country_profile['full_name']``."""
    want = str(country_full_name).strip().casefold()
    return series.astype(str).str.strip().str.casefold() == want


def _gcmt_status_weight(raw: Any) -> float:
    t = str(raw or "").strip().lower()
    if t == "operating":
        return 1.0
    if t in ("mothballed", "cancelled"):
        return 0.3
    if t in ("proposed", "shelved"):
        return 0.0
    return 0.0


def _read_lat_lon_size(df: pd.DataFrame) -> tuple[str, str, str | None]:
    cols = {_norm_col(c): str(c) for c in df.columns}
    lat_c = next((cols[k] for k in cols if k == "latitude"), None)
    lon_c = next((cols[k] for k in cols if k == "longitude"), None)
    size_c = None
    for k, orig in cols.items():
        kl = k.replace(" ", "")
        if "minesize" in kl or ("mine" in kl and "km2" in kl):
            size_c = orig
            break
    if lat_c is None or lon_c is None:
        raise ValueError(f"GEM coal sheet: need Latitude/Longitude; have {list(df.columns)[:20]}")
    return lat_c, lon_c, size_c


def _disk_radius_m(
    row: pd.Series,
    size_col: str | None,
    *,
    uniform_m: float | None,
    fallback_radius_m: float,
    size_radius_scale: float,
) -> float:
    if uniform_m is not None and float(uniform_m) > 0:
        return float(uniform_m)
    if not size_col or size_col not in row.index:
        return float(fallback_radius_m)
    try:
        km2 = float(row[size_col])
    except (TypeError, ValueError):
        return float(fallback_radius_m)
    if not np.isfinite(km2) or km2 <= 0:
        return float(fallback_radius_m)
    return float(np.sqrt(km2 * 1_000_000.0 / np.pi) * size_radius_scale)


def _burn_mask(
    gdf: gpd.GeoDataFrame,
    h: int,
    w: int,
    transform: Any,
    crs: Any,
    cams_cells: dict[int, dict[str, Any]],
) -> np.ndarray:
    fill = 0.0
    acc = np.zeros((h, w), dtype=np.float64)
    if gdf is None or gdf.empty:
        return acc.astype(np.float32)
    g = gdf.to_crs(crs)
    shapes = ((mapping(geom), 1.0) for geom in g.geometry if geom is not None and not geom.is_empty)
    features.rasterize(
        shapes,
        out_shape=(h, w),
        transform=transform,
        fill=fill,
        out=acc,
        dtype=np.float64,
        all_touched=True,
        merge_alg=MergeAlg.replace,
    )
    inside = pixels_inside_cams_cells(h, w, transform, crs, cams_cells)
    acc = np.where(inside, acc, fill)
    return (acc > 0.5).astype(np.float32)


def load_coal_mine_tracker_mask(
    xlsx_path: Path,
    country_full_name: str,
    h: int,
    w: int,
    transform: Any,
    crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
) -> np.ndarray:
    """Binary 0/1 on CAMS/CORINE grid: buffered GEM coal mine points (EPSG:3035 disks)."""
    non_name = str(cfg["non_closed_sheet"])
    closed_name = str(cfg["closed_sheet"])
    country_col = str(cfg["country_column"])
    closed_disk_m = float(cfg["closed_disk_m"])
    uniform_disk_m = cfg["uniform_disk_m"]
    apply_status_gate = bool(cfg["apply_status_gate"])
    fallback_radius_m = float(cfg["fallback_radius_m"])
    size_radius_scale = float(cfg["size_radius_scale"])

    non = pd.read_excel(xlsx_path, sheet_name=non_name)
    closed = pd.read_excel(xlsx_path, sheet_name=closed_name)
    non.columns = [str(c).replace("\n", " ").strip() for c in non.columns]
    closed.columns = [str(c).replace("\n", " ").strip() for c in closed.columns]
    if country_col not in non.columns or country_col not in closed.columns:
        raise ValueError(f"GEM coal: missing country column {country_col!r}")

    lat_n, lon_n, size_n = _read_lat_lon_size(non)
    lat_c, lon_c, _ = _read_lat_lon_size(closed)
    if "Mine Size (Km2)" in non.columns:
        size_n = "Mine Size (Km2)"

    um = float(uniform_disk_m) if uniform_disk_m is not None else None
    parts: list[gpd.GeoDataFrame] = []
    for df, is_closed in ((non, False), (closed, True)):
        if df.empty:
            continue
        m = _country_rows_eq(df[country_col], country_full_name)
        df = df.loc[m].copy()
        if df.empty:
            continue
        if apply_status_gate and "Status" in df.columns:
            df = df.assign(_gw=df["Status"].map(_gcmt_status_weight))
            df = df.loc[df["_gw"] > 0].copy()
        if df.empty:
            continue
        lat_col, lon_col = (lat_c, lon_c) if is_closed else (lat_n, lon_n)
        g = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col].astype(float), df[lat_col].astype(float), crs="EPSG:4326"),
            crs="EPSG:4326",
        )
        g = g.to_crs(crs)
        if is_closed:
            r = float(closed_disk_m)
            g["geometry"] = g.geometry.buffer(r)
        else:
            radii = [
                _disk_radius_m(
                    g.iloc[i],
                    size_n,
                    uniform_m=um,
                    fallback_radius_m=fallback_radius_m,
                    size_radius_scale=size_radius_scale,
                )
                for i in range(len(g))
            ]
            g["geometry"] = [g.geometry.iloc[i].buffer(radii[i]) for i in range(len(g))]
        parts.append(g[["geometry"]])

    if not parts:
        log.info(f"GEM coal: no rows for country {country_full_name!r}")
        return np.zeros((h, w), dtype=np.float32)
    allg = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=str(crs))
    return _burn_mask(allg, h, w, transform, crs, cams_cells)


def load_oil_gas_extractors_mask(
    xlsx_path: Path,
    country_full_name: str,
    h: int,
    w: int,
    transform: Any,
    crs: Any,
    cams_cells: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
) -> np.ndarray:
    """Binary 0/1: lat/lon points from the configured sheet, filtered by country/status, buffered in metres."""
    sheet = str(cfg["main_sheet"])
    country_col = str(cfg["country_column"])
    lat_col = str(cfg["lat_column"])
    lon_col = str(cfg["lon_column"])
    buf_default = float(cfg["point_buffer_m"])
    status_col = cfg["status_column"]
    status_allow = cfg["status_allow"]
    onshore_col = cfg["onshore_column"]
    buf_on = cfg["point_buffer_onshore_m"]
    buf_off = cfg["point_buffer_offshore_m"]

    main = pd.read_excel(xlsx_path, sheet_name=sheet)
    main.columns = [str(c).strip() for c in main.columns]
    if country_col not in main.columns:
        raise ValueError(f"GEM oil/gas sheet {sheet!r}: no column {country_col!r}; have {list(main.columns)[:25]}")
    main = main.loc[_country_rows_eq(main[country_col], country_full_name)].copy()
    if main.empty:
        log.info(f"GEM oil/gas: no rows for country {country_full_name!r}")
        return np.zeros((h, w), dtype=np.float32)

    if status_col is not None and str(status_col).strip():
        sc = str(status_col).strip()
        allow = {str(x).strip().lower() for x in (status_allow or [])}
        if sc not in main.columns:
            raise ValueError(f"GEM oil/gas: status column {sc!r} missing")
        main = main[main[sc].astype(str).str.strip().str.lower().isin(allow)].copy()
    if main.empty:
        log.info(f"GEM oil/gas: no rows after status filter ({country_full_name!r})")
        return np.zeros((h, w), dtype=np.float32)

    if lat_col not in main.columns or lon_col not in main.columns:
        raise ValueError(f"GEM oil/gas: need lat/lon columns {lat_col!r} {lon_col!r}")

    geoms = []
    crs_tgt = str(crs)
    use_onoff = (
        onshore_col is not None
        and str(onshore_col).strip()
        and str(onshore_col).strip() in main.columns
        and buf_on is not None
        and buf_off is not None
    )
    for _, row in main.iterrows():
        try:
            lat_f = float(row[lat_col])
            lon_f = float(row[lon_col])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(lat_f) and np.isfinite(lon_f)):
            continue
        if use_onoff:
            onv = str(row.get(str(onshore_col).strip(), "")).strip().lower()
            buf = float(buf_off) if onv == "offshore" else float(buf_on)
        else:
            buf = buf_default
        pt = gpd.GeoDataFrame(geometry=[Point(lon_f, lat_f)], crs="EPSG:4326").to_crs(crs_tgt)
        geoms.append(pt.geometry.iloc[0].buffer(buf))

    if not geoms:
        return np.zeros((h, w), dtype=np.float32)
    g = gpd.GeoDataFrame(geometry=geoms, crs=crs_tgt)
    return _burn_mask(g, h, w, transform, crs, cams_cells)
