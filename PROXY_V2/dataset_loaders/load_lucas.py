from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PROXY_V2.core import log
from PROXY_V2.core.raster_helpers import rasterize_buffered_points


def _as_list(x: Any) -> list:
    if isinstance(x, list):
        return list(x)
    return [x]


def _match_any_rules(group: Any) -> list[dict[str, Any]]:
    """One AND-group: either a single rule dict or a list of rule dicts."""
    if isinstance(group, dict):
        return [group]
    return list(group)


def _column_matches(series: pd.Series, value: Any) -> pd.Series:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return pd.to_numeric(series, errors="coerce") == float(value)
    want = {str(v).strip().upper() for v in _as_list(value)}
    got = series.astype(str).str.strip().str.upper()
    return got.isin(want)


def _collect_match_columns(signal_cfg: dict[str, Any]) -> set[str]:
    cols: set[str] = set()
    match_rules = signal_cfg.get("match")
    match_any = signal_cfg.get("match_any")
    for rules in ([match_rules] if match_rules else []) + (match_any or []):
        for rule in _match_any_rules(rules):
            if rule.get("column"):
                cols.add(str(rule["column"]))
    return cols


def _filter_lucas_df(df: pd.DataFrame, signal_cfg: dict[str, Any]) -> pd.DataFrame:
    match_rules = signal_cfg.get("match")
    match_any = signal_cfg.get("match_any")
    if not match_rules and not match_any:
        raise ValueError("LUCAS signal block needs 'match' or 'match_any'")

    if match_any:
        mask = pd.Series(False, index=df.index)
        for group in match_any:
            gmask = pd.Series(True, index=df.index)
            rules = _match_any_rules(group)
            for rule in rules:
                col = str(rule.get("column", "")).strip()
                val = rule.get("value")
                if not col or val is None:
                    raise ValueError(f"invalid LUCAS match rule: {rule!r}")
                gmask &= _column_matches(df[col], val)
            mask |= gmask
        n_rules = sum(len(_match_any_rules(g)) for g in match_any)
    else:
        mask = pd.Series(True, index=df.index)
        for rule in match_rules:
            col = str(rule.get("column", "")).strip()
            val = rule.get("value")
            if not col or val is None:
                raise ValueError(f"invalid LUCAS match rule: {rule!r}")
            mask &= _column_matches(df[col], val)
        n_rules = len(match_rules)

    log.debug(f"LUCAS filter: {int(mask.sum())} rows kept ({n_rules} rules)")
    return df.loc[mask]


def _read_lucas_country_df(
    lucas_path: Path,
    country_profile: dict[str, str],
    lucas_root_cfg: dict[str, Any],
    signal_cfgs: list[dict[str, Any]],
    *,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    country_col = str(lucas_root_cfg.get("country_column", "POINT_NUTS0"))
    lat_col = str(lucas_root_cfg.get("lat_column", "POINT_LAT"))
    lon_col = str(lucas_root_cfg.get("lon_column", "POINT_LONG"))
    cntr = str(country_profile["other"]).strip().upper()

    usecols = {country_col, lat_col, lon_col, *extra_columns}
    for sc in signal_cfgs:
        usecols |= _collect_match_columns(sc)

    df = pd.read_csv(lucas_path, usecols=lambda c: c in usecols, low_memory=False)
    log.debug(f"LUCAS read {len(df)} rows from {lucas_path.name}")

    cc = df[country_col].astype(str).str.strip().str.upper()
    df = df[cc == cntr].copy()
    log.info(f"LUCAS {cntr}: {len(df)} rows after country filter ({country_col})")
    return df


def load_lucas_points(
    lucas_path: Path,
    country_profile: dict[str, str],
    signal_cfg: dict[str, Any],
    *,
    lucas_root_cfg: dict[str, Any] | None = None,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    root = lucas_root_cfg or {}
    lat_col = str(root.get("lat_column", "POINT_LAT"))
    lon_col = str(root.get("lon_column", "POINT_LONG"))

    df = _read_lucas_country_df(
        lucas_path, country_profile, root, [signal_cfg], extra_columns=extra_columns
    )
    filtered = _filter_lucas_df(df, signal_cfg)
    if extra_columns:
        out = filtered[[lat_col, lon_col, *extra_columns]].copy()
        out.columns = ["lat", "lon", *extra_columns]
    else:
        out = filtered[[lat_col, lon_col]].copy()
        out.columns = ["lat", "lon"]
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"])
    log.info(f"LUCAS signal filter: {len(out)} points kept")
    return out


def rasterize_lucas_crop_groups(
    lucas_path: Path,
    country_profile: dict[str, str],
    lucas_root_cfg: dict[str, Any],
    group_signal_cfgs: dict[str, Any],
    crop_groups: list[str],
    *,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
    metric_crs: str,
    burn_value: float,
    all_touched: bool,
) -> dict[str, np.ndarray]:
    """One CSV read; buffered raster mask per crop group."""
    blocks = [group_signal_cfgs[g] for g in crop_groups]
    df = _read_lucas_country_df(lucas_path, country_profile, lucas_root_cfg, blocks)
    lat_col = str(lucas_root_cfg.get("lat_column", "POINT_LAT"))
    lon_col = str(lucas_root_cfg.get("lon_column", "POINT_LONG"))
    empty = np.zeros((int(height), int(width)), dtype=np.float32)
    masks: dict[str, np.ndarray] = {}

    for gname in crop_groups:
        block = group_signal_cfgs[gname]
        pts = _filter_lucas_df(df, block)[[lat_col, lon_col]].copy()
        pts[lat_col] = pd.to_numeric(pts[lat_col], errors="coerce")
        pts[lon_col] = pd.to_numeric(pts[lon_col], errors="coerce")
        pts = pts.dropna(subset=[lat_col, lon_col])
        if pts.empty:
            masks[gname] = empty.copy()
            log.info(f"LUCAS manure {gname}: 0 pts")
            continue
        masks[gname] = rasterize_buffered_points(
            pd.to_numeric(pts[lon_col], errors="coerce").to_numpy(),
            pd.to_numeric(pts[lat_col], errors="coerce").to_numpy(),
            buffer_m=float(block["buffer_m"]),
            metric_crs=metric_crs,
            height=int(height),
            width=int(width),
            transform=transform,
            raster_crs=raster_crs,
            burn_value=float(burn_value),
            fill=0.0,
            dtype=np.float32,
            all_touched=all_touched,
        )
        log.info(f"LUCAS manure {gname}: {len(pts)} pts sum={float(masks[gname].sum()):.4g}")
    return masks


def lucas_rate_mean_raster(
    pts: pd.DataFrame,
    buffer_m: float,
    metric_crs: str,
    *,
    height: int,
    width: int,
    transform: Any,
    raster_crs: Any,
) -> np.ndarray:
    """Mean Einarsson rate (kg N ha-1 yr-1) from buffered LUCAS points (sum/count rasterize)."""
    import geopandas as gpd
    from rasterio import features as rio_features
    from rasterio.enums import MergeAlg
    from shapely.geometry import Point, mapping

    h, w = int(height), int(width)
    out = np.full((h, w), np.nan, dtype=np.float32)
    if pts.empty:
        return out

    gdf = gpd.GeoDataFrame(
        {"rate": pts["rate"].astype(np.float64)},
        geometry=[Point(xy) for xy in zip(pts["lon"], pts["lat"])],
        crs="EPSG:4326",
    )
    gdf = gdf.to_crs(metric_crs)
    buf = float(buffer_m)
    if buf > 0:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(buf)
    gdf = gdf.to_crs(raster_crs)

    rate_sum = np.zeros((h, w), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.float32)
    shapes_r = [
        (mapping(g), float(r))
        for g, r in zip(gdf.geometry, gdf["rate"])
        if g is not None and not g.is_empty
    ]
    shapes_c = [
        (mapping(g), 1.0)
        for g in gdf.geometry
        if g is not None and not g.is_empty
    ]
    if not shapes_r:
        return out

    rio_features.rasterize(
        shapes_r,
        out=rate_sum,
        transform=transform,
        fill=0.0,
        dtype=np.float64,
        all_touched=True,
        merge_alg=MergeAlg.add,
    )
    rio_features.rasterize(
        shapes_c,
        out=cnt,
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        all_touched=True,
        merge_alg=MergeAlg.add,
    )
    hit = cnt > 0
    out[hit] = (rate_sum[hit] / cnt[hit]).astype(np.float32)
    log.info(f"LUCAS rate mean raster: {int(hit.sum())} px buffer={buf}m")
    return out
