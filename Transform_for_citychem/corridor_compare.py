"""Compare legacy vs new line emissions on two Athens road corridors."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lines_osm import G_S_TO_KG_YR, distribute_roads_gdf
from transform import (
    ROADS_SNAP,
    _repo_root,
    clip_area_rows,
    domain_box,
    load_config,
    load_config,
    parse_snap_map,
    read_grid_csv,
    reproject_xy,
    resolve_path,
    source_crs,
    to_area_rows,
)

ROOT = _repo_root()
CFG = load_config(Path(__file__).resolve().parent / "config.yaml")
OUT_DIR = ROOT / "Output/CityChem/Athens_2019/figures"
LEG_SHP = ROOT / "Output/OLD/ATHENS/Emissions/2019/V3/Increase_Factor_1/Results/lines.shp"
POLS = ("NOx", "NMVOC", "PM10")
MAJOR = {"motorway", "trunk", "primary"}


def load_new_gdf() -> gpd.GeoDataFrame:
    """Real OTM line geometry — not CSV bbox diagonals (xcor_* = segment bounds, not endpoints)."""
    root = _repo_root()
    input_dir = resolve_path(CFG["Input_folder"], root)
    epsg = int(CFG["EPSG"])
    step = int(CFG["Grid_step_m"])
    src_crs = source_crs(input_dir)
    sector = parse_snap_map(CFG["SNAP_TO_GNFR"])[ROADS_SNAP]
    area_df = reproject_xy(read_grid_csv(input_dir / sector / "area_emission_grid.csv"), src_crs, epsg)
    cells = clip_area_rows(to_area_rows(area_df, CFG, ROADS_SNAP, step), CFG)
    roads = CFG["Roads_lines"]
    gdf = distribute_roads_gdf(
        cells,
        resolve_path(roads["path"], root),
        layer=str(roads["layer"]),
        highway_column=str(roads["highway_column"]),
        dst_epsg=epsg,
        domain=domain_box(CFG),
    )
    return gdf


def leg_kg(row, pol: str) -> float:
    return float(row[pol]) * 1000.0


def new_kg(row, pol: str) -> float:
    return float(row[pol])


def pick_corridors(legacy: gpd.GeoDataFrame) -> list[tuple[str, object]]:
    major = legacy.loc[legacy.typeOfRoad.isin(MAJOR)].copy()
    major["cen_x"] = major.geometry.centroid.x
    major["cen_y"] = major.geometry.centroid.y
    major["NOx_kg"] = major["NOx"] * 1000.0

    c1 = major.loc[
        (major.typeOfRoad == "motorway")
        & (major.cen_x.between(714000, 722000))
        & (major.cen_y.between(4217000, 4226000))
    ]
    c2 = major.loc[
        (major.typeOfRoad.isin(["motorway", "trunk"]))
        & (major.cen_x.between(742000, 752000))
        & (major.cen_y.between(4214000, 4223000))
    ]
    corridors = []
    for name, sub in [
        ("Corridor A — west ring (motorway ~714–722k E)", c1),
        ("Corridor B — east radial (motorway/trunk ~742–752k E)", c2),
    ]:
        if sub.empty:
            raise RuntimeError(f"no segments for {name}")
        buf = unary_union(sub.geometry.values).buffer(800)
        corridors.append((name, buf))
    return corridors


def clip_to_corridor(gdf: gpd.GeoDataFrame, corridor) -> gpd.GeoDataFrame:
    hit = gdf[gdf.geometry.intersects(corridor)].copy()
    if hit.empty:
        return hit
    orig_geom = hit.geometry.copy()
    hit["geometry"] = hit.geometry.intersection(corridor)
    hit = hit.loc[~hit.geometry.is_empty].copy()
    orig_len = orig_geom.loc[hit.index].length
    new_len = hit.geometry.length
    ratio = np.divide(new_len, orig_len, out=np.zeros(len(hit)), where=orig_len > 0)
    for pol in POLS:
        hit[pol] = hit[pol].astype(float) * ratio
    return hit


def match_segments(leg: gpd.GeoDataFrame, new: gpd.GeoDataFrame, pol: str) -> pd.DataFrame:
    rows = []
    for i, lr in leg.iterrows():
        lgeom = lr.geometry
        lcent = lgeom.centroid
        dist = new.geometry.centroid.distance(lcent)
        j = int(dist.idxmin())
        nr = new.loc[j]
        rows.append({
            "legacy_idx": i,
            "new_idx": j,
            "match_m": float(dist.min()),
            "typeOfRoad": lr.typeOfRoad,
            "legacy_kg_yr": leg_kg(lr, pol),
            "new_kg_yr": new_kg(nr, pol),
            "length_m": float(lgeom.length),
        })
    return pd.DataFrame(rows)


def corridor_stats(name: str, leg: gpd.GeoDataFrame, new: gpd.GeoDataFrame) -> None:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(f"segments  legacy {len(leg)}  new {len(new)}")
    for pol in POLS:
        lk = sum(leg_kg(r, pol) for _, r in leg.iterrows())
        nk = sum(new_kg(r, pol) for _, r in new.iterrows())
        print(f"{pol:6s}  legacy {lk:10,.0f}  new {nk:10,.0f}  diff {(nk - lk) / lk * 100:+.1f}%")

    for pol in POLS:
        m = match_segments(leg, new, pol)
        m = m.loc[m.match_m <= 150]
        if len(m) < 5:
            print(f"{pol}: too few matched segments")
            continue
        lv = m.legacy_kg_yr.to_numpy()
        nv = m.new_kg_yr.to_numpy()
        r = float(np.corrcoef(lv, nv)[0, 1])
        rel = nv / np.maximum(lv, 1e-12)
        print(
            f"{pol} matched n={len(m)}  r={r:.3f}  "
            f"median new/legacy={np.median(rel):.3f}  mean={rel.mean():.3f}"
        )


def plot_corridor(name: str, leg: gpd.GeoDataFrame, new: gpd.GeoDataFrame, corridor, tag: str):
    pol = "NOx"
    m = match_segments(leg, new, pol)
    m = m.loc[m.match_m <= 150]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)

    corridor_g = gpd.GeoDataFrame(geometry=[corridor], crs=leg.crs)
    for ax, gdf, title, col in [
        (axes[0], leg, "legacy NOx (kg/yr)", "NOx"),
        (axes[1], new, "new NOx (kg/yr)", "NOx"),
    ]:
        corridor_g.boundary.plot(ax=ax, color="#333", linewidth=1.2, zorder=1)
        if not gdf.empty:
            vals = gdf[col].astype(float) * (1000.0 if gdf is leg else 1.0)
            gdf.assign(v=vals).plot(ax=ax, column="v", cmap="YlOrRd", linewidth=2.0, legend=True, zorder=2)
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[2]
    if not m.empty:
        ax.scatter(m.legacy_kg_yr, m.new_kg_yr, s=12, alpha=0.5, c="#4f7cff")
        mx = max(m.legacy_kg_yr.max(), m.new_kg_yr.max()) * 1.05
        ax.plot([0, mx], [0, mx], "k--", linewidth=0.8, label="1:1")
        ax.set_xlabel("legacy kg/yr")
        ax.set_ylabel("new kg/yr")
        r = np.corrcoef(m.legacy_kg_yr, m.new_kg_yr)[0, 1]
        ax.set_title(f"segment match NOx (r={r:.2f}, n={len(m)})", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle(name, fontsize=10)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"corridor_{tag}_NOx_Athens.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    legacy = gpd.read_file(LEG_SHP)
    new = load_new_gdf()
    corridors = pick_corridors(legacy)

    for i, (name, geom) in enumerate(corridors):
        leg = clip_to_corridor(legacy, geom)
        neu = clip_to_corridor(new, geom)
        corridor_stats(name, leg, neu)
        plot_corridor(name, leg, neu, geom, f"{i + 1}")


if __name__ == "__main__":
    main()
