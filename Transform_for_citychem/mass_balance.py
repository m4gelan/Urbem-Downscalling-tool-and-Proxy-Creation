"""Per-sector CAMS mass balance: expected (CAMS * in-domain weight) vs new vs legacy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from UrbEm_Visualizer.dataset_loaders.cams_emissions import load_cams_area_cells
from UrbEm_Visualizer.dataset_loaders.cams_grid import _cell_in_domain_envelope, domain_wgs84_from_domain
from UrbEm_Visualizer.dataset_loaders.countries import country_iso3
from UrbEm_Visualizer.dataset_loaders.tif_grid import (
    cell_id_on_raster,
    pixels_in_domain_bbox,
    read_weight_stack_native,
)
from UrbEm_Visualizer.downscaling.sector_meta import load_sector_yaml, sector_order, sector_mode
from UrbEm_Visualizer.downscaling.spatial import resolve_path
from UrbEm_Visualizer.paths import project_root
from UrbEm_Visualizer.pollutants import band_index_for_pollutant
from Transform_for_citychem.transform import read_grid_csv

POLLS = ("NOx", "NMVOC", "PM10")
STAT_COL = {"NMVOC": "NMVOC(tn/year)", "PM10": "PM10(tn/year)", "NOx": "NOX(tn/year)"}
SNAP_MAP = {
    "A_PublicPower": (1,), "C_OtherCombustion": (2,), "B_Industry": (3, 4),
    "D_Fugitive": (5,), "E_Solvents": (6,), "F_Roads": (7,),
    "J_Waste": (9,), "K_Agriculture": (10,), "I_Offroad": (12,),
}


def weight_fraction_in_domain(area_path: Path, domain: dict, cams_grid: dict, valid_cell_ids) -> dict[int, float]:
    """CAMS cell_id -> fraction of sector weights falling inside domain bbox."""
    labels, stack, tr, crs = read_weight_stack_native(area_path)
    bi = band_index_for_pollutant(labels, labels[0])
    w = stack[bi]
    h, w_px = int(w.shape[0]), int(w.shape[1])
    cell_id = cell_id_on_raster(tr, crs, h, w_px, cams_grid, valid_cell_ids)
    in_dom = pixels_in_domain_bbox(tr, crs, h, w_px, domain)
    out: dict[int, float] = {}
    flat_c = cell_id.ravel()
    flat_w = w.ravel().astype(np.float64)
    flat_in = in_dom.ravel()
    for cid in np.unique(flat_c[flat_c >= 0]):
        ic = int(cid)
        m = flat_c == ic
        s_all = float(flat_w[m].sum())
        s_dom = float(flat_w[m & flat_in].sum())
        if s_all > 0:
            out[ic] = s_dom / s_all
    return out


def main():
    run_cfg = yaml.safe_load(open(root / "UrbEm_Visualizer/config/run/Kozani_2019.yaml", encoding="utf-8"))
    domain = run_cfg["domain"]
    cams_nc = project_root() / run_cfg["paths"]["cams"]
    country = run_cfg["country"]
    iso3 = country_iso3(country)
    w, s, e, n = domain_wgs84_from_domain(domain)
    urbem = root / "Output/UrbEm/Kozani_2019"
    legacy_dir = root / "Output/OLD/KOZANI/Emissions/2019/V3/Increase_Factor_1/Results/Results_CSVs"
    tag = "Nasia_Kozani_CAMS_v3_1"
    stat_areas = pd.read_csv(legacy_dir / f"{tag}_urbem_stat_areas_sources_2019.csv", sep=";")

    rows = []
    for sid in sector_order(run_cfg):
        if sector_mode(sid) not in ("both", "area_only"):
            continue
        sec_yaml = load_sector_yaml(sid)
        cams_area = sec_yaml.get("cams_area_sources")
        if not cams_area:
            continue

        sec_cfg = run_cfg["sectors"][sid]
        aw = sec_cfg.get("area_weights") or {}
        aw_path = aw.get("path")
        if not aw_path:
            continue
        area_tif = resolve_path(str(aw_path), project_root())

        cells, cams_grid = load_cams_area_cells(
            cams_nc,
            year=int(cams_area["year"]),
            country_iso3=iso3,
            emission_category_indices=list(cams_area["emission_category_indices"]),
            source_type_indices=list(cams_area["source_type_indices"]),
            pollutants=list(POLLS),
        )
        valid = frozenset(cells.keys()) if cells else frozenset()
        wf = weight_fraction_in_domain(area_tif, domain, cams_grid, valid)

        sd = urbem / sid
        new_area = {p: 0.0 for p in POLLS}
        if sd.is_dir():
            df = read_grid_csv(sd / "area_emission_grid.csv")
            for p in POLLS:
                sub = df.loc[df["pollutant"] == p]
                new_area[p] += float(sub["emission"].sum()) if not sub.empty else 0.0

        legacy_area = {p: 0.0 for p in POLLS}
        for snap in SNAP_MAP.get(sid, ()):
            sub = stat_areas.loc[stat_areas["SNAP"] == snap]
            for p in POLLS:
                col = STAT_COL[p]
                if col in sub.columns:
                    legacy_area[p] += float(sub[col].sum()) * 1000.0

        overlap = [cid for cid, row in cells.items() if _cell_in_domain_envelope(row, w, s, e, n)]
        for p in POLLS:
            cams_full = sum(float(cells[c]["pollutants_within_cell"].get(p, 0.0)) for c in overlap)
            cams_expected = sum(
                float(cells[c]["pollutants_within_cell"].get(p, 0.0)) * wf.get(c, 0.0)
                for c in overlap
            )
            no_weight = [c for c in overlap if wf.get(c, 0.0) <= 0.0]
            cams_no_proxy = sum(float(cells[c]["pollutants_within_cell"].get(p, 0.0)) for c in no_weight)
            rows.append({
                "sector": sid,
                "pollutant": p,
                "n_overlap_cells": len(overlap),
                "n_zero_weight_cells": len(no_weight),
                "cams_full_kg": cams_full,
                "cams_no_proxy_kg": cams_no_proxy,
                "cams_expected_kg": cams_expected,
                "new_area_kg": new_area[p],
                "legacy_area_kg": legacy_area[p],
                "new_vs_expected_pct": new_area[p] / cams_expected * 100 if cams_expected else None,
                "legacy_vs_expected_pct": legacy_area[p] / cams_expected * 100 if cams_expected else None,
                "new_loss_vs_expected_kg": cams_expected - new_area[p],
                "legacy_loss_vs_expected_kg": cams_expected - legacy_area[p],
            })

    df = pd.DataFrame(rows)
    out = root / "Output/CityChem/Kozani_2019/figures/mass_balance_by_sector.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}\n")

    for p in POLLS:
        sub = df[df["pollutant"] == p]
        print(f"=== {p} (area sectors) ===")
        print(f"  CAMS full overlap:     {sub['cams_full_kg'].sum():,.0f} kg")
        print(f"  CAMS no proxy weight:  {sub['cams_no_proxy_kg'].sum():,.0f} kg  (cells with zero weight in domain)")
        print(f"  CAMS expected in dom:  {sub['cams_expected_kg'].sum():,.0f} kg  (full * weight fraction)")
        print(f"  New downscaling:       {sub['new_area_kg'].sum():,.0f} kg  ({sub['new_area_kg'].sum()/sub['cams_expected_kg'].sum()*100:.1f}% of expected)")
        print(f"  Legacy stat:           {sub['legacy_area_kg'].sum():,.0f} kg  ({sub['legacy_area_kg'].sum()/sub['cams_expected_kg'].sum()*100:.1f}% of expected)")
        print(f"  New loss vs expected:  {sub['new_loss_vs_expected_kg'].sum():,.0f} kg")
        print(f"  Legacy loss vs expected: {sub['legacy_loss_vs_expected_kg'].sum():,.0f} kg")
        print()

    print("=== Per-sector NMVOC: where mass is lost vs CAMS expected ===")
    sub = df[df["pollutant"] == "NMVOC"].sort_values("legacy_loss_vs_expected_kg", ascending=False)
    for _, r in sub.iterrows():
        print(
            f"  {r['sector']:18s}  expected={r['cams_expected_kg']:9,.0f}  "
            f"new={r['new_area_kg']:9,.0f} ({r['new_vs_expected_pct']:5.1f}%)  "
            f"legacy={r['legacy_area_kg']:9,.0f} ({r['legacy_vs_expected_pct']:5.1f}%)  "
            f"no_proxy={r['cams_no_proxy_kg']:7,.0f}  zero_w_cells={int(r['n_zero_weight_cells'])}"
        )


if __name__ == "__main__":
    main()
