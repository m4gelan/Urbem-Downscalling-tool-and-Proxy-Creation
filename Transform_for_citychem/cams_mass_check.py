"""Compare downscaling totals vs CAMS cell masses over the run domain."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from UrbEm_Visualizer.dataset_loaders.cams_emissions import load_cams_area_cells
from UrbEm_Visualizer.dataset_loaders.cams_grid import _cell_in_domain_envelope, domain_wgs84_from_domain
from UrbEm_Visualizer.dataset_loaders.countries import country_iso3
from UrbEm_Visualizer.downscaling.sector_meta import load_sector_yaml, sector_order, sector_mode
from Transform_for_citychem.transform import read_grid_csv

POLLS = ("NOx", "NMVOC", "PM10")
STAT_COL = {"NMVOC": "NMVOC(tn/year)", "PM10": "PM10(tn/year)", "NOx": "NOX(tn/year)"}
SNAP_MAP = {
    "A_PublicPower": (1,),
    "C_OtherCombustion": (2,),
    "B_Industry": (3, 4),
    "D_Fugitive": (5,),
    "E_Solvents": (6,),
    "F_Roads": (7,),
    "G_Shipping": (8,),
    "H_Aviation": (8,),
    "I_Offroad": (8,),
    "J_Waste": (9,),
    "K_Agriculture": (10,),
}


def main():
    run_cfg = yaml.safe_load(open(root / "UrbEm_Visualizer/config/run/Kozani_2019.yaml", encoding="utf-8"))
    domain = run_cfg["domain"]
    cams_nc = root / run_cfg["paths"]["cams"]
    country = run_cfg["country"]
    year = int(run_cfg["year"])
    iso3 = country_iso3(country)
    w, s, e, n = domain_wgs84_from_domain(domain)
    urbem = root / "Output/UrbEm/Kozani_2019"
    legacy_dir = root / "Output/OLD/KOZANI/Emissions/2019/V3/Increase_Factor_1/Results/Results_CSVs"
    tag = "Nasia_Kozani_CAMS_v3_1"
    stat_areas = pd.read_csv(legacy_dir / f"{tag}_urbem_stat_areas_sources_{year}.csv", sep=";")
    stat_lines = pd.read_csv(legacy_dir / f"{tag}_urbem_stat_lines_sources_{year}.csv", sep=";")

    clip_frac = {}
    clip_path = urbem / "clip_log.json"
    if clip_path.is_file():
        for row in json.loads(clip_path.read_text(encoding="utf-8")):
            clip_frac[(row["sector"], row["pollutant"], row["cell_id"])] = row["clipped_mass_fraction"]

    rows = []
    for sid in sector_order(run_cfg):
        mode = sector_mode(sid)
        sec_yaml = load_sector_yaml(sid)
        cams_area = sec_yaml.get("cams_area_sources")

        cams_full = {p: 0.0 for p in POLLS}
        cams_prorated = {p: 0.0 for p in POLLS}
        n_cells = 0

        if cams_area and mode in ("both", "area_only"):
            cells, _ = load_cams_area_cells(
                cams_nc,
                year=int(cams_area["year"]),
                country_iso3=iso3,
                emission_category_indices=list(cams_area["emission_category_indices"]),
                source_type_indices=list(cams_area["source_type_indices"]),
                pollutants=list(POLLS),
            )
            for cid, row in cells.items():
                if not _cell_in_domain_envelope(row, w, s, e, n):
                    continue
                n_cells += 1
                for p in POLLS:
                    m = float(row["pollutants_within_cell"].get(p, 0.0))
                    cams_full[p] += m
                    frac = clip_frac.get((sid, p, cid), 1.0)
                    cams_prorated[p] += m * frac

        new_area = {p: 0.0 for p in POLLS}
        new_point = {p: 0.0 for p in POLLS}
        sd = urbem / sid
        if sd.is_dir():
            for kind, out in (("area", new_area), ("point", new_point)):
                pth = sd / f"{kind}_emission_grid.csv"
                if not pth.is_file() or pth.stat().st_size <= 30:
                    continue
                df = read_grid_csv(pth)
                for p in POLLS:
                    sub = df.loc[df["pollutant"] == p]
                    out[p] += float(sub["emission"].sum()) if not sub.empty else 0.0

        legacy_area = {p: 0.0 for p in POLLS}
        legacy_line = {p: 0.0 for p in POLLS}
        for snap in SNAP_MAP.get(sid, ()):
            sub_a = stat_areas.loc[stat_areas["SNAP"] == snap]
            sub_l = stat_lines.loc[stat_lines["SNAP"] == snap]
            for p in POLLS:
                col = STAT_COL[p]
                if col in sub_a.columns:
                    legacy_area[p] += float(sub_a[col].sum()) * 1000.0
                if col in sub_l.columns:
                    legacy_line[p] += float(sub_l[col].sum()) * 1000.0

        for p in POLLS:
            new_total = new_area[p] + new_point[p]
            legacy_total = legacy_area[p] + legacy_line[p]
            rows.append({
                "sector": sid,
                "pollutant": p,
                "n_cams_cells": n_cells,
                "cams_full_kg": cams_full[p],
                "cams_prorated_kg": cams_prorated[p],
                "new_area_kg": new_area[p],
                "new_point_kg": new_point[p],
                "new_total_kg": new_total,
                "legacy_area_kg": legacy_area[p],
                "legacy_line_kg": legacy_line[p],
                "legacy_total_kg": legacy_total,
            })

    df = pd.DataFrame(rows)
    for num in df.select_dtypes("number").columns:
        if num != "n_cams_cells":
            df[f"{num}_pct_diff_new_vs_cams_full"] = (df["new_total_kg"] / df["cams_full_kg"] - 1.0) * 100.0
            df[f"pct_diff_new_vs_legacy"] = (df["new_total_kg"] / df["legacy_total_kg"] - 1.0) * 100.0
            df[f"pct_diff_legacy_vs_cams_full"] = (df["legacy_total_kg"] / df["cams_full_kg"] - 1.0) * 100.0

    out = root / "Output/CityChem/Kozani_2019/figures/cams_mass_diagnostic.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}")

    print("\n=== Totals by pollutant (all sectors) ===")
    for p in POLLS:
        sub = df[df["pollutant"] == p]
        cf = sub["cams_full_kg"].sum()
        cp = sub["cams_prorated_kg"].sum()
        nt = sub["new_total_kg"].sum()
        lt = sub["legacy_total_kg"].sum()
        print(f"\n{p}:")
        print(f"  CAMS full (overlap cells):  {cf:,.0f} kg/yr")
        print(f"  CAMS prorated (clip log):   {cp:,.0f} kg/yr")
        print(f"  New downscaling:            {nt:,.0f} kg/yr  ({nt/cf*100-100:+.1f}% vs CAMS full)" if cf else f"  New downscaling: {nt:,.0f}")
        print(f"  Legacy urbem_stat:          {lt:,.0f} kg/yr  ({lt/cf*100-100:+.1f}% vs CAMS full)" if cf else f"  Legacy: {lt:,.0f}")

    print("\n=== Area-only sectors (excl. F_Roads, H_Aviation) — typical ~20% pattern ===")
    area_sids = [s for s in sector_order(run_cfg) if s not in ("F_Roads", "H_Aviation")]
    for p in POLLS:
        sub = df[(df["pollutant"] == p) & (df["sector"].isin(area_sids))]
        cf = sub["cams_full_kg"].sum()
        nt = sub["new_area_kg"].sum()
        lt = sub["legacy_area_kg"].sum()
        if cf <= 0:
            continue
        print(f"{p}: new_area={nt:,.0f} legacy_area={lt:,.0f} cams_full={cf:,.0f}  "
              f"new/cams={nt/cf*100:.1f}% legacy/cams={lt/cf*100:.1f}%")

    print("\n=== Per-sector NMVOC (area path) ===")
    sub = df[(df["pollutant"] == "NMVOC") & (df["cams_full_kg"] > 0)]
    for _, r in sub.iterrows():
        pct_new = (r["new_total_kg"] / r["cams_full_kg"] - 1) * 100 if r["cams_full_kg"] else float("nan")
        pct_leg = (r["legacy_total_kg"] / r["cams_full_kg"] - 1) * 100 if r["cams_full_kg"] else float("nan")
        print(f"  {r['sector']:18s}  cams={r['cams_full_kg']:9,.0f}  new={r['new_total_kg']:9,.0f} ({pct_new:+.1f}%)  "
              f"legacy={r['legacy_total_kg']:9,.0f} ({pct_leg:+.1f}%)")


if __name__ == "__main__":
    main()
