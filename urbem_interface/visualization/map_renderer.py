"""
Map rendering utilities - convert domain and data to GeoJSON for Leaflet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pyproj import Transformer
from rasterio.crs import CRS

NA_VAL = -999.0


def domain_bounds_wgs84_for_map(domain_cfg: dict) -> tuple[float, float, float, float]:
    """
    Project domain bounds to WGS84 (lon, lat).
    Returns (lon_min, lat_min, lon_max, lat_max).
    """
    xmin, ymin, xmax, ymax = (
        domain_cfg["xmin"],
        domain_cfg["ymin"],
        domain_cfg["xmax"],
        domain_cfg["ymax"],
    )
    domain_crs = CRS.from_string(domain_cfg["crs"])
    to_wgs = Transformer.from_crs(domain_crs, CRS.from_epsg(4326), always_xy=True)
    xs = [xmin, xmin, xmax, xmax]
    ys = [ymin, ymax, ymin, ymax]
    lons, lats = to_wgs.transform(xs, ys)
    return (float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats)))


def domain_to_geojson(domain_cfg: dict) -> dict[str, Any]:
    """
    Convert domain to GeoJSON polygon for Leaflet.
    """
    lon_min, lat_min, lon_max, lat_max = domain_bounds_wgs84_for_map(domain_cfg)
    coords = [
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ]
    return {
        "type": "Feature",
        "properties": {"name": "Domain"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }


def domain_to_geojson_str(domain_cfg: dict) -> str:
    """Return domain GeoJSON as JSON string."""
    return json.dumps(domain_to_geojson(domain_cfg))


def output_snaps_and_pollutants(output_path: str | Path) -> dict[str, Any]:
    """
    Read output CSV and return available pollutants and SNAPs per pollutant.
    For area sources only.
    """
    path = Path(output_path)
    if not path.exists():
        return {"pollutants": [], "snap_by_pollutant": {}}
    df = pd.read_csv(path)
    poll_cols = [c for c in df.columns if c in {"NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"}]
    if not poll_cols or "snap" not in df.columns:
        return {"pollutants": poll_cols or ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"], "snap_by_pollutant": {}}
    snap_by_pollutant: dict[str, list[int]] = {}
    for p in poll_cols:
        if p not in df.columns:
            continue
        non_zero = df[df[p] > 0]
        snaps = sorted(non_zero["snap"].dropna().astype(int).unique().tolist())
        if snaps:
            snap_by_pollutant[p] = snaps
    return {"pollutants": poll_cols, "snap_by_pollutant": snap_by_pollutant}


def output_to_geojson(
    output_path: str | Path,
    source_type: str,
    domain_cfg: dict,
    pollutant: str | None = None,
    mode: str = "total",
    snap_id: int | None = None,
) -> dict[str, Any]:
    """
    Read output CSV and convert to GeoJSON for map display.
    Coordinates are in domain CRS and are reprojected to WGS84.
    """
    path = Path(output_path)
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}

    domain_crs = CRS.from_string(domain_cfg["crs"])
    to_wgs = Transformer.from_crs(domain_crs, CRS.from_epsg(4326), always_xy=True)

    df = pd.read_csv(path)

    if source_type == "line":
        if "ycor_star" in df.columns and "ycor_start" not in df.columns:
            df = df.rename(columns={"ycor_star": "ycor_start"})
        for col in ["xcor_start", "ycor_start", "xcor_end", "ycor_end"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["xcor_start", "ycor_start", "xcor_end", "ycor_end"])
        df = df[(df["xcor_start"] > NA_VAL) & (df["ycor_start"] > NA_VAL)]

        poll_col = pollutant or ("NOx" if "NOx" in df.columns else None)
        if not poll_col or poll_col not in df.columns:
            poll_col = next((c for c in ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"] if c in df.columns), None)

        features = []
        for _, row in df.iterrows():
            x0, y0 = float(row["xcor_start"]), float(row["ycor_start"])
            x1, y1 = float(row["xcor_end"]), float(row["ycor_end"])
            lon0, lat0 = to_wgs.transform(x0, y0)
            lon1, lat1 = to_wgs.transform(x1, y1)
            val = float(row[poll_col]) if poll_col and poll_col in row else 0
            props = {"value": val, poll_col: val} if poll_col and val > 0 else {}
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon0, lat0], [lon1, lat1]],
                },
            })
        return {"type": "FeatureCollection", "features": features}

    if source_type == "point":
        for col in ["xcor", "ycor"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["xcor", "ycor"])
        df = df[(df["xcor"] > NA_VAL) & (df["ycor"] > NA_VAL)]

        features = []
        for _, row in df.iterrows():
            x, y = float(row["xcor"]), float(row["ycor"])
            lon, lat = to_wgs.transform(x, y)
            features.append({
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            })
        return {"type": "FeatureCollection", "features": features}

    if source_type == "area":
        for col in ["xcor_sw", "ycor_sw", "xcor_ne", "ycor_ne"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["xcor_sw", "ycor_sw", "xcor_ne", "ycor_ne"])
        df = df[(df["xcor_sw"] > NA_VAL) & (df["ycor_sw"] > NA_VAL)]

        poll_col = pollutant or ("NOx" if "NOx" in df.columns else None)
        if not poll_col or poll_col not in df.columns:
            poll_col = next((c for c in ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"] if c in df.columns), None)

        if mode == "snap" and snap_id is not None and "snap" in df.columns:
            df = df[df["snap"].astype(int) == snap_id].copy()
        elif mode == "total" and "snap" in df.columns:
            poll_cols_agg = [c for c in df.columns if c in {"NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"}]
            if poll_cols_agg:
                df = df.groupby(["xcor_sw", "ycor_sw", "xcor_ne", "ycor_ne"], as_index=False)[poll_cols_agg].sum()

        features = []
        for _, row in df.iterrows():
            x1, y1 = float(row["xcor_sw"]), float(row["ycor_sw"])
            x2, y2 = float(row["xcor_ne"]), float(row["ycor_ne"])
            lon1, lat1 = to_wgs.transform(x1, y1)
            lon2, lat2 = to_wgs.transform(x2, y2)
            coords = [
                [lon1, lat1], [lon2, lat1], [lon2, lat2], [lon1, lat2], [lon1, lat1]
            ]
            val = float(row[poll_col]) if poll_col and poll_col in row else 0
            props = {"value": val, poll_col: val} if poll_col and val > 0 else {}
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords],
                },
            })
        return {"type": "FeatureCollection", "features": features}

    return {"type": "FeatureCollection", "features": []}


def _raster_cell_to_polygon(
    row: int,
    col: int,
    nrow: int,
    ncol: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> list[list[float]]:
    """Return cell corners in (x, y) domain coords."""
    cell_w = (xmax - xmin) / ncol
    cell_h = (ymax - ymin) / nrow
    x1 = xmin + col * cell_w
    y1 = ymax - (row + 1) * cell_h
    x2 = x1 + cell_w
    y2 = y1 + cell_h
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]


def raster_csv_to_geojson(
    csv_path: str | Path,
    domain_cfg: dict,
    value_column: str = "value",
) -> dict[str, Any]:
    """
    Convert raster CSV (row, col, value_columns...) to GeoJSON.
    CSV has row, col and one or more value columns. Use value_column to pick which.
    """
    path = Path(csv_path)
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}

    df = pd.read_csv(path)
    if "row" not in df.columns or "col" not in df.columns:
        return {"type": "FeatureCollection", "features": []}

    if value_column not in df.columns:
        value_column = next((c for c in df.columns if c not in ("row", "col")), "value")
    if value_column not in df.columns:
        return {"type": "FeatureCollection", "features": []}

    nrow = int(domain_cfg["nrow"])
    ncol = int(domain_cfg["ncol"])
    xmin = float(domain_cfg["xmin"])
    ymin = float(domain_cfg["ymin"])
    xmax = float(domain_cfg["xmax"])
    ymax = float(domain_cfg["ymax"])
    domain_crs = CRS.from_string(domain_cfg["crs"])
    to_wgs = Transformer.from_crs(domain_crs, CRS.from_epsg(4326), always_xy=True)

    df["row"] = pd.to_numeric(df["row"], errors="coerce").astype(int)
    df["col"] = pd.to_numeric(df["col"], errors="coerce").astype(int)
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df = df[(df["row"] >= 0) & (df["row"] < nrow) & (df["col"] >= 0) & (df["col"] < ncol)]

    features = []
    for _, rec in df.iterrows():
        r, c = int(rec["row"]), int(rec["col"])
        val = float(rec[value_column])
        if val <= NA_VAL or (val != val):
            continue
        corners = _raster_cell_to_polygon(r, c, nrow, ncol, xmin, ymin, xmax, ymax)
        lons, lats = to_wgs.transform([p[0] for p in corners], [p[1] for p in corners])
        coords = [[float(lons[i]), float(lats[i])] for i in range(len(corners))]
        features.append({
            "type": "Feature",
            "properties": {"value": val},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
    return {"type": "FeatureCollection", "features": features}


def raster_sum_to_geojson(
    csv_paths: list[str | Path],
    domain_cfg: dict,
    value_column: str = "value",
) -> dict[str, Any]:
    """
    Read multiple raster CSVs, sum by (row, col), convert to GeoJSON.
    Used for CAMS total and downscaled total.
    """
    if not csv_paths:
        return {"type": "FeatureCollection", "features": []}
    dfs = []
    for p in csv_paths:
        path = Path(p)
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "row" not in df.columns or "col" not in df.columns:
            continue
        if value_column not in df.columns:
            continue
        dfs.append(df[["row", "col", value_column]])
    if not dfs:
        return {"type": "FeatureCollection", "features": []}
    combined = pd.concat(dfs, ignore_index=True)
    combined["row"] = pd.to_numeric(combined["row"], errors="coerce").astype(int)
    combined["col"] = pd.to_numeric(combined["col"], errors="coerce").astype(int)
    combined[value_column] = pd.to_numeric(combined[value_column], errors="coerce").fillna(0)
    summed = combined.groupby(["row", "col"], as_index=False)[value_column].sum()
    summed = summed[summed[value_column] > 0]
    return raster_df_to_geojson(summed, domain_cfg, value_column)


def raster_df_to_geojson(
    df: pd.DataFrame,
    domain_cfg: dict,
    value_column: str = "value",
) -> dict[str, Any]:
    """Convert a dataframe with row, col, value to GeoJSON."""
    if df.empty or "row" not in df.columns or "col" not in df.columns or value_column not in df.columns:
        return {"type": "FeatureCollection", "features": []}
    nrow = int(domain_cfg["nrow"])
    ncol = int(domain_cfg["ncol"])
    xmin = float(domain_cfg["xmin"])
    ymin = float(domain_cfg["ymin"])
    xmax = float(domain_cfg["xmax"])
    ymax = float(domain_cfg["ymax"])
    domain_crs = CRS.from_string(domain_cfg["crs"])
    to_wgs = Transformer.from_crs(domain_crs, CRS.from_epsg(4326), always_xy=True)
    df = df[(df["row"] >= 0) & (df["row"] < nrow) & (df["col"] >= 0) & (df["col"] < ncol)]
    features = []
    for _, rec in df.iterrows():
        r, c = int(rec["row"]), int(rec["col"])
        val = float(rec[value_column])
        if val <= NA_VAL or (val != val):
            continue
        corners = _raster_cell_to_polygon(r, c, nrow, ncol, xmin, ymin, xmax, ymax)
        lons, lats = to_wgs.transform([p[0] for p in corners], [p[1] for p in corners])
        coords = [[float(lons[i]), float(lats[i])] for i in range(len(corners))]
        features.append({
            "type": "Feature",
            "properties": {"value": val},
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
    return {"type": "FeatureCollection", "features": features}


def cams_grid_to_geojson(
    intermediates_dir: str | Path,
    domain_cfg: dict,
) -> dict[str, Any]:
    """
    Build CAMS coarse grid as GeoJSON from step2_coarse_grid.
    """
    base = Path(intermediates_dir)
    meta_path = base / "step2_coarse_grid" / "cams_origin_metadata.csv"
    if not meta_path.exists():
        return {"type": "FeatureCollection", "features": []}

    meta = pd.read_csv(meta_path).iloc[0]
    nrow = int(meta["nrow"])
    ncol = int(meta["ncol"])
    xmin = float(meta["xmin"])
    ymin = float(meta["ymin"])
    xmax = float(meta["xmax"])
    ymax = float(meta["ymax"])
    crs_str = str(meta.get("crs", domain_cfg.get("crs", "EPSG:32634")))

    domain_crs = CRS.from_string(crs_str)
    to_wgs = Transformer.from_crs(domain_crs, CRS.from_epsg(4326), always_xy=True)

    cell_w = (xmax - xmin) / ncol
    cell_h = (ymax - ymin) / nrow

    features = []
    for r in range(nrow):
        for c in range(ncol):
            x1 = xmin + c * cell_w
            y1 = ymax - (r + 1) * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]]
            lons, lats = to_wgs.transform([p[0] for p in corners], [p[1] for p in corners])
            coords = [[float(lons[i]), float(lats[i])] for i in range(len(corners))]
            features.append({
                "type": "Feature",
                "properties": {"row": r, "col": c, "cell_id": r * ncol + c + 1},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            })
    return {"type": "FeatureCollection", "features": features}
