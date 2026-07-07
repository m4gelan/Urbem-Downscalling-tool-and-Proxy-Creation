from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

POLLUTANTS = ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"]
LINE_COLS = ["snap", "xcor_start", "ycor_start", "xcor_end", "ycor_end", "elevation", "width", *POLLUTANTS]
LINE_COLS_WITH_CATEGORY = ["snap", "f_category", *LINE_COLS[1:]]
ROAD_GROUPS = ("motorway", "trunk", "primary", "secondary")
ROAD_WEIGHTS = {"motorway": 10, "trunk": 5, "primary": 2, "secondary": 2}
WIDTH_BY_TYPE = {
    "motorway": 20, "motorway_link": 20,
    "trunk": 16, "trunk_link": 16,
    "primary": 12, "primary_link": 12,
    "secondary": 12, "secondary_link": 12,
}
KG_YR_TO_G_S = 1000.0 / (365.0 * 24.0 * 3600.0)
G_S_TO_KG_YR = 1.0 / KG_YR_TO_G_S

def _normalize_highway(raw: str) -> str | None:
    v = str(raw).strip()
    if v in ("motorway", "motorway_link"):
        return "motorway"
    if v in ("trunk", "trunk_link"):
        return "trunk"
    if v in ("primary", "primary_link"):
        return "primary"
    if v in ("secondary", "secondary_link"):
        return "secondary"
    return None


def _assign_width(raw: str) -> int:
    return int(WIDTH_BY_TYPE.get(str(raw).strip(), 12))


def _grid_from_cells(cells: pd.DataFrame) -> gpd.GeoDataFrame:
    rows = []
    for i, row in cells.reset_index(drop=True).iterrows():
        geom = box(float(row["xcor_sw"]), float(row["ycor_sw"]), float(row["xcor_ne"]), float(row["ycor_ne"]))
        rec = {
            "grid_index": int(i),
            "xcor_sw": int(row["xcor_sw"]),
            "ycor_sw": int(row["ycor_sw"]),
            "xcor_ne": int(row["xcor_ne"]),
            "ycor_ne": int(row["ycor_ne"]),
        }
        for pol in POLLUTANTS:
            rec[pol] = float(row[pol])
        rows.append({**rec, "geometry": geom})
    return gpd.GeoDataFrame(rows, crs=f"EPSG:{cells.attrs.get('epsg', 32634)}")


def _clip_lines_to_domain(
    gdf: gpd.GeoDataFrame,
    domain: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    xmin, ymin, xmax, ymax = domain
    clip_geom = box(xmin, ymin, xmax, ymax)
    out = gdf.copy()
    orig_len = out.geometry.length.to_numpy()
    out["geometry"] = out.geometry.intersection(clip_geom)
    new_len = out.geometry.length.to_numpy()
    ok = new_len > 0
    out = out.loc[ok].copy()
    ratio = np.divide(new_len[ok], orig_len[ok], out=np.zeros(int(ok.sum())), where=orig_len[ok] > 0)
    for pol in POLLUTANTS:
        out[pol] = out[pol].astype(float).to_numpy() * ratio
    return out.reset_index(drop=True)


def distribute_roads_to_lines(
    cells: pd.DataFrame,
    roads_path: Path,
    *,
    layer: str,
    highway_column: str,
    dst_epsg: int,
    domain: tuple[float, float, float, float],
    snap: int = 7,
    f_category: str | None = None,
) -> pd.DataFrame:
    cols = LINE_COLS_WITH_CATEGORY if f_category is not None else LINE_COLS
    gdf = distribute_roads_gdf(
        cells, roads_path, layer=layer, highway_column=highway_column,
        dst_epsg=dst_epsg, domain=domain, snap=snap,
    )
    if gdf.empty:
        return pd.DataFrame(columns=cols)
    out = gdf.copy()
    seg_bounds = out.geometry.bounds
    out["xcor_start"] = seg_bounds["minx"].astype(int)
    out["ycor_start"] = seg_bounds["miny"].astype(int)
    out["xcor_end"] = seg_bounds["maxx"].astype(int)
    out["ycor_end"] = seg_bounds["maxy"].astype(int)
    if f_category is not None:
        out["f_category"] = f_category
    for pol in POLLUTANTS:
        out[pol] = out[pol] * KG_YR_TO_G_S
    return out.loc[out[POLLUTANTS].sum(axis=1) > 0, cols].reset_index(drop=True)


def distribute_roads_gdf(
    cells: pd.DataFrame,
    roads_path: Path,
    *,
    layer: str,
    highway_column: str,
    dst_epsg: int,
    domain: tuple[float, float, float, float],
    snap: int = 7,
) -> gpd.GeoDataFrame:
    """Return road segments with pollutant columns in kg/yr and real line geometry."""
    cols = ["snap", "elevation", "width", *POLLUTANTS, "geometry"]
    if cells.empty:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs=f"EPSG:{dst_epsg}")

    cells = cells.copy()
    cells.attrs["epsg"] = dst_epsg
    grid = _grid_from_cells(cells)

    roads = gpd.read_file(roads_path, layer=layer, mask=grid)
    if roads.crs is None:
        raise ValueError(f"{roads_path}: missing CRS")
    roads = roads.to_crs(grid.crs)
    if highway_column not in roads.columns:
        raise ValueError(f"{roads_path} layer {layer!r}: missing {highway_column!r}")

    roads = roads.copy()
    roads["Highway"] = roads[highway_column].map(_normalize_highway)
    roads = roads.loc[roads["Highway"].notna()].copy()
    roads["length"] = roads.geometry.length
    roads = roads.loc[roads["length"] > 0].copy()

    joined = gpd.sjoin(roads, grid[["grid_index", "geometry"]], how="inner", predicate="intersects")
    if joined.empty:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs=grid.crs)

    parts: list[gpd.GeoDataFrame] = []
    totals = {pol: float(cells[pol].sum()) for pol in POLLUTANTS}

    for grid_index in joined["grid_index"].unique():
        cell = grid.loc[grid["grid_index"] == grid_index].iloc[0]
        road_cell = joined.loc[joined["grid_index"] == grid_index].copy()
        present = [g for g in ROAD_GROUPS if (road_cell["Highway"] == g).any()]
        if not present:
            continue

        weight_sum = sum(ROAD_WEIGHTS[g] for g in present)
        for group in present:
            segs = road_cell.loc[road_cell["Highway"] == group].copy()
            seg_len = float(segs["length"].sum())
            if seg_len <= 0:
                continue
            share = ROAD_WEIGHTS[group] / weight_sum
            out = segs[["geometry", "length", highway_column]].copy()
            for pol in POLLUTANTS:
                out[pol] = segs["length"] / seg_len * share * float(cell[pol])
            parts.append(out)

    if not parts:
        return gpd.GeoDataFrame(columns=cols, geometry=[], crs=grid.crs)

    out = pd.concat(parts, ignore_index=True)
    for pol in POLLUTANTS:
        s = float(out[pol].sum())
        if s > 0:
            out[pol] = out[pol] * (totals[pol] / s)

    out["width"] = out[highway_column].map(_assign_width)
    out["snap"] = int(snap)
    out["elevation"] = 0
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=grid.crs)
    return _clip_lines_to_domain(out, domain)
