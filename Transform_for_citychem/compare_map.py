"""Legacy vs new emission maps by SNAP — separate figures for area, line, point."""

from __future__ import annotations

from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from shapely.geometry import Point, box

from lines_osm import G_S_TO_KG_YR, distribute_roads_gdf
from transform import (
    ROADS_SNAP,
    _repo_root,
    clip_area_rows,
    domain_box,
    km_sw,
    iter_sector_snaps,
    load_config,
    read_grid_csv,
    reproject_xy,
    resolve_path,
    parse_snap_map,
    source_crs,
    to_area_rows,
)

POLLUTANTS = ("NMVOC", "PM10")
SOURCE_KINDS = ("area", "line", "point")
LEGACY_INDUSTRY_SNAPS = (3, 4)
LEGACY_STAT_SNAP = {
    "A_PublicPower": (1,),
    "C_OtherCombustion": (2,),
    "B_Industry": LEGACY_INDUSTRY_SNAPS,
    "D_Fugitive": (5,),
    "E_Solvents": (6,),
    "G_Shipping": (8,),
    "H_Aviation": (11,),
    "I_Offroad": (12,),
    "J_Waste": (9,),
    "K_Agriculture": (10,),
}


def legacy_delivery_snaps(internal_snap: int, sector: str) -> tuple[int, ...]:
    if sector in LEGACY_STAT_SNAP:
        return LEGACY_STAT_SNAP[sector]
    if internal_snap == 3:
        return LEGACY_INDUSTRY_SNAPS
    return (internal_snap,)


def legacy_csv_dir(cfg: dict, root: Path) -> Path:
    city = str(cfg["City"])
    year = int(cfg["Year"])
    return root / "Output" / "OLD" / city.upper() / "Emissions" / str(year) / "V3" / "Increase_Factor_1" / "Results" / "Results_CSVs"


def legacy_tag(cfg: dict) -> str:
    return f"Nasia_{cfg['City']}_CAMS_v3_1"


def _empty(crs: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"v": pd.Series(dtype=float)}, geometry=[], crs=crs)


def _area_gdf(df: pd.DataFrame, pol: str, crs: str) -> gpd.GeoDataFrame:
    if df.empty or pol not in df.columns:
        return _empty(crs)
    df = df.loc[df[pol] > 0].copy()
    if df.empty:
        return _empty(crs)
    geoms = [box(r.xcor_sw, r.ycor_sw, r.xcor_ne, r.ycor_ne) for r in df.itertuples()]
    return gpd.GeoDataFrame({"v": df[pol].astype(float).values}, geometry=geoms, crs=crs)


def _point_gdf(df: pd.DataFrame, pol: str, crs: str) -> gpd.GeoDataFrame:
    if df.empty or pol not in df.columns:
        return _empty(crs)
    df = df.loc[df[pol] > 0].copy()
    if df.empty:
        return _empty(crs)
    geoms = [Point(r.xcor, r.ycor) for r in df.itertuples()]
    return gpd.GeoDataFrame({"v": df[pol].astype(float).values}, geometry=geoms, crs=crs)


def _sector_area_1km(sector_dir: Path, pol: str, src_crs: str, epsg: int, step: int) -> gpd.GeoDataFrame:
    df = reproject_xy(read_grid_csv(sector_dir / "area_emission_grid.csv"), src_crs, epsg)
    sub = df.loc[df["pollutant"] == pol].copy()
    if sub.empty:
        return _empty(f"EPSG:{epsg}")
    sub["xcor_sw"] = sub["x"].map(lambda v: km_sw(v, step))
    sub["ycor_sw"] = sub["y"].map(lambda v: km_sw(v, step))
    agg = sub.groupby(["xcor_sw", "ycor_sw"], as_index=False)["emission"].sum()
    agg["xcor_ne"] = agg["xcor_sw"] + step
    agg["ycor_ne"] = agg["ycor_sw"] + step
    return _area_gdf(agg.rename(columns={"emission": pol}), pol, f"EPSG:{epsg}")


def _sector_point(sector_dir: Path, pol: str, src_crs: str, epsg: int) -> gpd.GeoDataFrame:
    df = reproject_xy(read_grid_csv(sector_dir / "point_emission_grid.csv"), src_crs, epsg)
    sub = df.loc[df["pollutant"] == pol].copy()
    if sub.empty:
        return _empty(f"EPSG:{epsg}")
    wide = sub.groupby(["x", "y"], as_index=False)["emission"].sum()
    return _point_gdf(wide.rename(columns={"x": "xcor", "y": "ycor", "emission": pol}), pol, f"EPSG:{epsg}")


def _road_cells(cfg, sector_dir, internal_snap, src_crs, epsg, step):
    area_df = reproject_xy(read_grid_csv(sector_dir / "area_emission_grid.csv"), src_crs, epsg)
    return clip_area_rows(to_area_rows(area_df, cfg, internal_snap, step), cfg)


def _lines_gdf(cells, roads_path, layer, highway_column, epsg, domain, pol, scale_total: float | None = None):
    gdf = distribute_roads_gdf(
        cells, roads_path, layer=layer, highway_column=highway_column,
        dst_epsg=epsg, domain=domain,
    )
    if gdf.empty or pol not in gdf.columns:
        return _empty(f"EPSG:{epsg}")
    vals = gdf[pol].astype(float)
    if scale_total is not None and vals.sum() > 0:
        vals = vals * (scale_total / vals.sum())
    return gpd.GeoDataFrame({"v": vals.values}, geometry=gdf.geometry.values, crs=gdf.crs)


def load_legacy_gdf(
    kind: str,
    legacy_dir: Path,
    tag: str,
    year: int,
    internal_snap: int,
    sector: str,
    pol: str,
    epsg: int,
    domain,
    road_cells,
    roads_path,
    roads_layer,
    roads_hwy,
) -> gpd.GeoDataFrame:
    crs = f"EPSG:{epsg}"
    legacy_snaps = legacy_delivery_snaps(internal_snap, sector)
    if kind == "area":
        areas = pd.read_csv(legacy_dir / f"{tag}_areas_sources_{year}.csv")
        areas = areas.loc[areas["snap"].isin(legacy_snaps)]
        return _area_gdf(areas, pol, crs)

    if kind == "point":
        points = pd.read_csv(legacy_dir / f"{tag}_point_sources_{year}.csv")
        if points.empty:
            return _empty(crs)
        points = points.loc[points["snap"].isin(legacy_snaps)]
        return _point_gdf(points, pol, crs)

    if kind == "line" and internal_snap == ROADS_SNAP:
        lines = pd.read_csv(legacy_dir / f"{tag}_lines_sources_{year}_all_increase.csv")
        leg_total = float(lines[pol].sum()) * G_S_TO_KG_YR if pol in lines.columns else 0.0
        # legacy CSV stores g/s but all coords are 0 — map totals onto OTM geometry
        return _lines_gdf(road_cells, roads_path, roads_layer, roads_hwy, epsg, domain, pol, scale_total=leg_total)

    return _empty(crs)


def load_new_gdf(
    kind: str,
    sector_dir: Path,
    internal_snap: int,
    pol: str,
    src_crs: str,
    epsg: int,
    step: int,
    cfg,
    domain,
    road_cells,
    roads_path,
    roads_layer,
    roads_hwy,
) -> gpd.GeoDataFrame:
    crs = f"EPSG:{epsg}"
    if kind == "area" and internal_snap != ROADS_SNAP:
        return _sector_area_1km(sector_dir, pol, src_crs, epsg, step)
    if kind == "point" and internal_snap != ROADS_SNAP:
        return _sector_point(sector_dir, pol, src_crs, epsg)
    if kind == "line" and internal_snap == ROADS_SNAP:
        return _lines_gdf(road_cells, roads_path, roads_layer, roads_hwy, epsg, domain, pol)
    return _empty(crs)


def _bounds(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return None
    b = gdf.total_bounds
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def _draw_gdf(ax, gdf: gpd.GeoDataFrame, *, kind: str, title: str, vmin: float, vmax: float, crs: str):
    if gdf.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=8)
        return

    b = _bounds(gdf)
    pad_x = (b[2] - b[0]) * 0.05 or 1000.0
    pad_y = (b[3] - b[1]) * 0.05 or 1000.0
    ax.set_xlim(b[0] - pad_x, b[2] + pad_x)
    ax.set_ylim(b[1] - pad_y, b[3] + pad_y)
    ax.set_aspect("equal")
    norm = LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, vmin * 1.01))

    if kind == "area":
        gdf.plot(ax=ax, column="v", cmap="YlOrRd", norm=norm, alpha=0.65, linewidth=0.2, zorder=2)
    elif kind == "line":
        gdf.plot(ax=ax, column="v", cmap="YlOrRd", norm=norm, linewidth=1.4, zorder=3)
    else:
        sizes = np.clip(gdf["v"] / vmax * 100.0, 10.0, 100.0)
        gdf.plot(ax=ax, color="#7b2cbf", markersize=sizes, alpha=0.85, zorder=4)

    try:
        cx.add_basemap(ax, crs=crs, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")
    except Exception:
        ax.set_facecolor("#eef1f5")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def _vmax(gdfs: list[gpd.GeoDataFrame]) -> float:
    vals = []
    for g in gdfs:
        if not g.empty:
            vals.extend(g["v"].astype(float).tolist())
    return float(max(vals)) if vals else 1.0


def plot_kind(cfg: dict, root: Path, pol: str, kind: str, snaps: list[tuple[int, str]], out_path: Path):
    epsg = int(cfg["EPSG"])
    step = int(cfg["Grid_step_m"])
    year = int(cfg["Year"])
    crs = f"EPSG:{epsg}"
    input_dir = resolve_path(cfg["Input_folder"], root)
    legacy_dir = legacy_csv_dir(cfg, root)
    tag = legacy_tag(cfg)
    src_crs = source_crs(input_dir)
    domain = domain_box(cfg)
    roads = cfg["Roads_lines"]
    roads_path = resolve_path(roads["path"], root)
    roads_layer = str(roads["layer"])
    roads_hwy = str(roads["highway_column"])

    roads_sector_dir = input_dir / roads_sector_folder(cfg)
    road_cells = _road_cells(cfg, roads_sector_dir, ROADS_SNAP, src_crs, epsg, step) if roads_sector_dir.is_dir() else None

    panels = []
    for internal_snap, sector in snaps:
        sector_dir = input_dir / sector
        rc = road_cells if internal_snap == ROADS_SNAP else None
        leg = load_legacy_gdf(
            kind, legacy_dir, tag, year, internal_snap, sector, pol, epsg, domain,
            rc, roads_path, roads_layer, roads_hwy,
        )
        new = load_new_gdf(
            kind, sector_dir, internal_snap, pol, src_crs, epsg, step, cfg, domain,
            rc, roads_path, roads_layer, roads_hwy,
        )
        label = f"SNAP_{internal_snap} {sector}"
        panels.append((label, leg, new))

    panels = [(label, leg, new) for label, leg, new in panels if not (leg.empty and new.empty)]
    if not panels:
        print(f"skip {out_path} (no data)")
        return

    vmax = _vmax([g for _, l, n in panels for g in (l, n)])
    vmin = max(vmax * 1e-4, 1e-3)
    n = len(panels)
    fig, axes = plt.subplots(n, 2, figsize=(10, max(2.4 * n, 3.5)), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])

    for i, (label, leg, new) in enumerate(panels):
        note = ""
        if kind == "line" and i == 0:
            note = "\n(legacy: OTM geom, legacy totals)"
        _draw_gdf(axes[i, 0], leg, kind=kind, title=f"{label} legacy{note}", vmin=vmin, vmax=vmax, crs=crs)
        _draw_gdf(axes[i, 1], new, kind=kind, title=f"{label} new", vmin=vmin, vmax=vmax, crs=crs)

    fig.suptitle(f"{pol} — {kind} sources (kg/yr)", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def roads_sector_folder(cfg) -> str:
    folder = parse_snap_map(cfg["SNAP_TO_GNFR"]).get(ROADS_SNAP)
    if not folder:
        raise KeyError(f"SNAP_TO_GNFR missing SNAP_{ROADS_SNAP}")
    return folder


def main():
    root = _repo_root()
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = load_config(cfg_path)
    sector_snaps = iter_sector_snaps(cfg["SNAP_TO_GNFR"])
    input_dir = resolve_path(cfg["Input_folder"], root)
    city = str(cfg["City"])

    snaps = [(s, sec) for s, sec in sector_snaps if (input_dir / sec).is_dir()]
    fig_dir = resolve_path(cfg["Output_folder"], root) / "figures"

    for pol in POLLUTANTS:
        for kind in SOURCE_KINDS:
            plot_kind(cfg, root, pol, kind, snaps, fig_dir / f"compare_{kind}_{pol}_{city}.png")


if __name__ == "__main__":
    main()
