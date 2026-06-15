"""Per-SNAP mass totals (kg/yr) — legacy internal stat vs new sector grids."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lines_osm import G_S_TO_KG_YR, distribute_roads_gdf
from transform import (
    ROADS_SNAP,
    _repo_root,
    clip_area_rows,
    domain_box,
    iter_sector_snaps,
    load_config,
    read_grid_csv,
    reproject_xy,
    resolve_path,
    source_crs,
    to_area_rows,
)

REPORT_POLS = ("NMVOC", "PM10", "NOx")
STAT_COL = {"NMVOC": "NMVOC(tn/year)", "PM10": "PM10(tn/year)", "NOx": "NOX(tn/year)"}
LEGACY_INDUSTRY_SNAPS = (3, 4)
# legacy urbem_stat SNAP rows (pre-2026 export still used SNAP 11/12 for H/I)
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


def legacy_csv_dir(cfg: dict, root: Path) -> Path:
    city = str(cfg["City"])
    year = int(cfg["Year"])
    return root / "Output" / "OLD" / city.upper() / "Emissions" / str(year) / "V3" / "Increase_Factor_1" / "Results" / "Results_CSVs"


def legacy_tag(cfg: dict) -> str:
    return f"Nasia_{cfg['City']}_CAMS_v3_1"


def _load_stat(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def _stat_kg(stat: pd.DataFrame, snaps: tuple[int, ...], pol: str) -> float:
    col = STAT_COL[pol]
    sub = stat.loc[stat["SNAP"].isin(snaps)]
    if sub.empty or col not in sub.columns:
        return 0.0
    return float(sub[col].sum()) * 1000.0


def _grid_mass(sector_dir: Path, src_crs: str, epsg: int, pol: str, kind: str) -> float:
    path = sector_dir / f"{kind}_emission_grid.csv"
    df = reproject_xy(read_grid_csv(path), src_crs, epsg)
    sub = df.loc[df["pollutant"] == pol]
    return float(sub["emission"].sum()) if not sub.empty else 0.0


def legacy_area_for_sector(sector: str, stat_areas, pol: str) -> float:
    snaps = LEGACY_STAT_SNAP.get(sector)
    if not snaps:
        return 0.0
    return _stat_kg(stat_areas, snaps, pol)


def legacy_internal_mass(stat_areas, stat_lines, sector: str, internal_snap, pol):
    if internal_snap == ROADS_SNAP:
        return 0.0, _stat_kg(stat_lines, (ROADS_SNAP,), pol), 0.0
    area = legacy_area_for_sector(sector, stat_areas, pol)
    return area, 0.0, 0.0


def new_internal_mass(cfg, sector_dir, internal_snap, pol, src_crs, epsg, step, roads_path, roads_layer, roads_hwy):
    if internal_snap == ROADS_SNAP:
        area_df = reproject_xy(read_grid_csv(sector_dir / "area_emission_grid.csv"), src_crs, epsg)
        cells = clip_area_rows(to_area_rows(area_df, cfg, internal_snap, step), cfg)
        gdf = distribute_roads_gdf(
            cells, roads_path, layer=roads_layer, highway_column=roads_hwy,
            dst_epsg=epsg, domain=domain_box(cfg),
        )
        line = float(gdf[pol].sum()) if pol in gdf.columns and not gdf.empty else 0.0
        return 0.0, line, 0.0
    area = _grid_mass(sector_dir, src_crs, epsg, pol, "area")
    point = _grid_mass(sector_dir, src_crs, epsg, pol, "point")
    return area, 0.0, point


def legacy_csv_snap_mass(areas, lines, points, snap, pol, sector=None):
    if sector and sector in LEGACY_STAT_SNAP:
        area_snaps = LEGACY_STAT_SNAP[sector]
    elif snap == 3:
        area_snaps = LEGACY_INDUSTRY_SNAPS
    else:
        area_snaps = (snap,)
    la = float(areas.loc[areas["snap"].isin(area_snaps), pol].sum()) if areas["snap"].isin(area_snaps).any() else 0.0
    ll = 0.0
    if snap == ROADS_SNAP:
        ll = float(lines.loc[lines["snap"] == ROADS_SNAP, pol].sum()) * G_S_TO_KG_YR
        la = 0.0
    lp = 0.0
    if not points.empty and snap in points["snap"].values:
        lp = float(points.loc[points["snap"] == snap, pol].sum())
    return la, ll, lp


def main():
    root = _repo_root()
    cfg = load_config(Path(__file__).resolve().parent / "config.yaml")
    sector_snaps = iter_sector_snaps(cfg["SNAP_TO_GNFR"])
    input_dir = resolve_path(cfg["Input_folder"], root)
    out_dir = resolve_path(cfg["Output_folder"], root)
    legacy_dir = legacy_csv_dir(cfg, root)
    tag = legacy_tag(cfg)
    year = int(cfg["Year"])
    city = str(cfg["City"])
    epsg = int(cfg["EPSG"])
    step = int(cfg["Grid_step_m"])
    src_crs = source_crs(input_dir)
    roads = cfg["Roads_lines"]
    roads_path = resolve_path(roads["path"], root)

    stat_areas = _load_stat(legacy_dir / f"{tag}_urbem_stat_areas_sources_{year}.csv")
    stat_lines = _load_stat(legacy_dir / f"{tag}_urbem_stat_lines_sources_{year}.csv")
    areas = pd.read_csv(legacy_dir / f"{tag}_areas_sources_{year}.csv")
    points = pd.read_csv(legacy_dir / f"{tag}_point_sources_{year}.csv")
    lines = pd.read_csv(legacy_dir / f"{tag}_lines_sources_{year}_all_increase.csv")
    new_areas = pd.read_csv(out_dir / f"area_source_{city}.csv")
    new_lines = pd.read_csv(out_dir / f"line_source_{city}.csv")
    new_points = pd.read_csv(out_dir / f"point_source_{city}.csv")

    rows = []
    for internal_snap, sector in sector_snaps:
        sector_dir = input_dir / sector
        if not sector_dir.is_dir():
            continue
        for pol in REPORT_POLS:
            la, ll, lp = legacy_internal_mass(stat_areas, stat_lines, sector, internal_snap, pol)
            na, nl, np_ = new_internal_mass(
                cfg, sector_dir, internal_snap, pol, src_crs, epsg, step,
                roads_path, roads["layer"], roads["highway_column"],
            )
            c_la, c_ll, c_lp = legacy_csv_snap_mass(areas, lines, points, internal_snap, pol, sector)
            note = ""
            if internal_snap == 3:
                note = "legacy stat = SNAP 3+4 (v1 industry split)"
            elif internal_snap == 8:
                note = "export SNAP 8: shipping + aviation + offroad"
            elif internal_snap == 10:
                note = "export SNAP 10: K_AgriLivestock + L_AgriOther (single K_Agriculture folder)"
            lt = la + ll + lp
            nt = na + nl + np_
            diff_pct = (nt - lt) / lt * 100.0 if lt > 1e-9 else float("nan")
            rows.append({
                "snap": internal_snap,
                "sector": sector,
                "pollutant": pol,
                "legacy_area_kg_yr": la,
                "legacy_line_kg_yr": ll,
                "legacy_point_kg_yr": lp,
                "legacy_total_kg_yr": lt,
                "new_area_kg_yr": na,
                "new_line_kg_yr": nl,
                "new_point_kg_yr": np_,
                "new_total_kg_yr": nt,
                "diff_pct": diff_pct,
                "legacy_csv_area_kg_yr": c_la,
                "note": note,
            })

    df = pd.DataFrame(rows)
    out_path = out_dir / "figures" / f"mass_compare_{city}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    print("\n--- per-SNAP comparison (legacy urbem_stat vs new sector) ---")
    for pol in REPORT_POLS:
        sub = df[df["pollutant"] == pol]
        print(f"{pol}: legacy {sub.legacy_total_kg_yr.sum():.0f}  new {sub.new_total_kg_yr.sum():.0f}  "
              f"diff {(sub.new_total_kg_yr.sum()-sub.legacy_total_kg_yr.sum())/sub.legacy_total_kg_yr.sum()*100:.1f}%")

    print("\n--- CSV totals (legacy delivery vs transformed output) ---")
    for pol in REPORT_POLS:
        la = float(areas[pol].sum())
        ll = float(lines[pol].sum()) * G_S_TO_KG_YR
        lp = float(points[pol].sum()) if not points.empty else 0.0
        na = float(new_areas[pol].sum())
        nl = float(new_lines[pol].sum()) * G_S_TO_KG_YR
        np_ = float(new_points[pol].sum()) if not new_points.empty else 0.0
        print(f"{pol}: legacy {la+ll+lp:.0f}  new {na+nl+np_:.0f}  "
              f"diff {(na+nl+np_-la-ll-lp)/(la+ll+lp)*100:.1f}%")


if __name__ == "__main__":
    main()
