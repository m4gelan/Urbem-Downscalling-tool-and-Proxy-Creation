#!/usr/bin/env python3
"""
Map LUCAS 2022 survey points in Greece: agricultural LC1/LU1.

Uses Agriculture/aux_lucas_lc1_mapping.LUCAS_LC1_TO_CROP for legend labels
(category key from synthetic N table, plus LC1 code).

Default: one PNG per LC1 "block" (B11–B19 cereals, B21–B23 roots, …, E grassland).

Usage (from project root):
  python Agriculture/Auxiliaries/map_lucas_greece_lc_lu.py
  python Agriculture/Auxiliaries/map_lucas_greece_lc_lu.py --out-dir Agriculture/results/plots/lucas_greece_lc_blocks
  python Agriculture/Auxiliaries/map_lucas_greece_lc_lu.py --overview

Requires: pandas, geopandas, matplotlib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_crop_mapping():
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from Agriculture.aux_lucas_lc1_mapping import LUCAS_LC1_TO_CROP, crop_category_from_lc1

    return LUCAS_LC1_TO_CROP, crop_category_from_lc1


# LC1 blocks aligned with aux_lucas_lc1_mapping.LUCAS_LC1_TO_CROP
LC1_BLOCK_GROUPS: list[tuple[str, str, frozenset[str]]] = [
    ("cereals", "Cereals (B11–B19)", frozenset(f"B1{i}" for i in range(1, 10))),
    ("root_crops", "Root crops (B21–B23)", frozenset(["B21", "B22", "B23"])),
    ("industrial", "Industrial crops (B31–B37)", frozenset(f"B3{i}" for i in range(1, 8))),
    ("vegetables", "Vegetables and fodder on arable (B41–B45)", frozenset(f"B4{i}" for i in range(1, 6))),
    ("fodder_legumes", "Fodder legumes and temporary grass (B51–B55)", frozenset(f"B5{i}" for i in range(1, 6))),
    ("fruit_tree", "Permanent fruit and tree crops (B71–B77)", frozenset(f"B7{i}" for i in range(1, 8))),
    ("other_b8", "Olive, vineyard, nurseries, other (B81–B84)", frozenset(f"B8{i}" for i in range(1, 5))),
    ("fallow_bx", "Fallow (BX1–BX2)", frozenset(["BX1", "BX2"])),
    ("grassland", "Grassland (E10–E30)", frozenset(["E10", "E20", "E30"])),
]


def legend_label_lc1(lc1: str, lu_map: dict[str, str]) -> str:
    """Legend line: synthetic-N category (LC1), from mapping."""
    lc = str(lc1).strip().upper()
    cat = lu_map.get(lc)
    if cat is None:
        return lc
    return f"{cat} ({lc})"


def load_greece_ag_points(lucas_csv: Path) -> "pd.DataFrame":
    import pandas as pd

    usecols = [
        "POINT_ID",
        "POINT_NUTS0",
        "POINT_NUTS2",
        "POINT_LAT",
        "POINT_LONG",
        "SURVEY_LC1",
        "SURVEY_LU1",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        lucas_csv,
        usecols=lambda c: c in usecols,
        chunksize=300_000,
        dtype=str,
        low_memory=False,
    ):
        chunk = chunk[chunk["POINT_NUTS0"].astype(str).str.strip().str.upper() == "EL"].copy()
        if chunk.empty:
            continue
        chunk["LC1"] = chunk["SURVEY_LC1"].astype(str).str.strip().str.upper()
        chunk["LU1"] = chunk["SURVEY_LU1"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["LC1"].str.len() > 0]
        chunk = chunk[chunk["LC1"].str[0].isin(["B", "E"])]
        chunk["lat"] = pd.to_numeric(chunk["POINT_LAT"], errors="coerce")
        chunk["lon"] = pd.to_numeric(chunk["POINT_LONG"], errors="coerce")
        chunk = chunk.dropna(subset=["lat", "lon"])
        parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _plot_panel(
    gdf: "gpd.GeoDataFrame",
    greece_wgs: "gpd.GeoDataFrame",
    *,
    title: str,
    subtitle: str,
    color_col: str,
    out_png: Path,
    dpi: int,
    marker_size: float,
    legend_title: str,
) -> None:
    import matplotlib.pyplot as plt

    cats = sorted(gdf[color_col].unique())
    try:
        tab20 = plt.colormaps["tab20"]
    except AttributeError:
        tab20 = plt.cm.get_cmap("tab20")
    color_map = {c: tab20(i % 20) for i, c in enumerate(cats)}

    fig, ax = plt.subplots(figsize=(11, 12))
    greece_wgs.boundary.plot(ax=ax, color="0.35", linewidth=0.8)
    greece_wgs.plot(ax=ax, color="whitesmoke", edgecolor="0.35", linewidth=0.6)

    for cat in cats:
        sub = gdf[gdf[color_col] == cat]
        ax.scatter(
            sub.geometry.x,
            sub.geometry.y,
            s=marker_size,
            c=[color_map[cat]],
            label=f"{cat} (n={len(sub)})",
            alpha=0.88,
            linewidths=0.25,
            edgecolors="0.25",
        )

    b = greece_wgs.total_bounds
    ax.set_xlim(b[0] - 0.45, b[2] + 0.45)
    ax.set_ylim(b[1] - 0.35, b[3] + 0.55)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=6,
        framealpha=0.92,
        title=legend_title,
    )
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close()


def run(
    lucas_csv: Path,
    nuts_gpkg: Path,
    out_dir: Path,
    *,
    overview_path: Path | None,
    combo_mode: bool,
    top_n_combo: int,
    dpi: int,
    marker_size: float,
) -> None:
    import geopandas as gpd

    LUCAS_LC1_TO_CROP, _ = _import_crop_mapping()

    df = load_greece_ag_points(lucas_csv)
    if df.empty:
        raise SystemExit("No agricultural LUCAS points found for Greece (EL).")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )

    nuts = gpd.read_file(nuts_gpkg)
    el0 = nuts[(nuts["CNTR_CODE"].astype(str).str.upper() == "EL") & (nuts["LEVL_CODE"] == 0)]
    if el0.empty:
        raise SystemExit("Greece country boundary not found in NUTS gpkg (EL, LEVL_CODE 0).")
    greece_wgs = el0.dissolve().to_crs(4326)

    if combo_mode:
        gdf["lc_lu"] = gdf["LC1"].astype(str) + " | " + gdf["LU1"].astype(str)
        vc = gdf["lc_lu"].value_counts()
        top = set(vc.head(top_n_combo).index)
        gdf["plot_cat"] = gdf["lc_lu"].where(gdf["lc_lu"].isin(top), other="other")
        color_col = "plot_cat"
        title = "LUCAS 2022 — Greece — LC1|LU1 (top combinations)"
        subtitle = f"n = {len(gdf)} points"
        _plot_panel(
            gdf,
            greece_wgs,
            title=title,
            subtitle=subtitle,
            color_col=color_col,
            out_png=out_dir / "lucas_greece_lc_lu_combo.png",
            dpi=dpi,
            marker_size=marker_size,
            legend_title="LC1 | LU1",
        )
        print(f"Wrote {out_dir / 'lucas_greece_lc_lu_combo.png'}")
        return

    gdf["legend_label"] = gdf["LC1"].map(lambda x: legend_label_lc1(x, LUCAS_LC1_TO_CROP))

    for slug, block_title, lc_set in LC1_BLOCK_GROUPS:
        sub = gdf[gdf["LC1"].isin(lc_set)].copy()
        if sub.empty:
            print(f"Skip (no points): {block_title}")
            continue
        fname = f"lucas_greece_{slug}.png"
        out_png = out_dir / fname
        _plot_panel(
            sub,
            greece_wgs,
            title=f"LUCAS 2022 — Greece — {block_title}",
            subtitle=f"n = {len(sub)} points · legend = synthetic-N category (LC1)",
            color_col="legend_label",
            out_png=out_png,
            dpi=dpi,
            marker_size=marker_size,
            legend_title="Category (LC1)",
        )
        print(f"Wrote {out_png} ({len(sub)} points)")

    if overview_path is not None:
        _plot_panel(
            gdf,
            greece_wgs,
            title="LUCAS 2022 — Greece — all agricultural B/E (overview)",
            subtitle=f"n = {len(gdf)} points · legend = synthetic-N category (LC1)",
            color_col="legend_label",
            out_png=overview_path,
            dpi=dpi,
            marker_size=marker_size,
            legend_title="Category (LC1)",
        )
        print(f"Wrote {overview_path} ({len(gdf)} points)")


def main(argv: list[str] | None = None) -> int:
    root = _project_root()
    p = argparse.ArgumentParser(
        description="Map Greece LUCAS agricultural LC1/LU1 using crop mapping legends; one map per LC1 block."
    )
    p.add_argument(
        "--lucas",
        type=Path,
        default=root / "data" / "Agriculture" / "EU_LUCAS_2022.csv",
        help="Path to EU LUCAS CSV",
    )
    p.add_argument(
        "--nuts",
        type=Path,
        default=root / "Data" / "geometry" / "NUTS_RG_20M_2021_3035.gpkg",
        help="NUTS GeoPackage (for Greece outline)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=root / "Agriculture" / "results" / "plots" / "lucas_greece_lc_blocks",
        help="Directory for per-block PNG maps",
    )
    p.add_argument(
        "--overview",
        action="store_true",
        help="Also write a single overview map with all B/E points (same legend)",
    )
    p.add_argument(
        "--overview-path",
        type=Path,
        default=None,
        help="Path for overview PNG (default: out-dir/../lucas_greece_lc_lu_overview.png)",
    )
    p.add_argument(
        "--combo",
        action="store_true",
        help="Instead of block maps, write one combo map (LC1|LU1 top-N)",
    )
    p.add_argument("--top", type=int, default=18, help="Top N LC1|LU1 combos when --combo")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument(
        "--size",
        type=float,
        default=36.0,
        help="Matplotlib scatter marker area (s)",
    )
    args = p.parse_args(argv)

    if not args.lucas.is_file():
        print(f"LUCAS file not found: {args.lucas}", file=sys.stderr)
        return 1
    if not args.nuts.is_file():
        alt = root / "data" / "geometry" / "NUTS_RG_20M_2021_3035.gpkg"
        if alt.is_file():
            args.nuts = alt
        else:
            print(f"NUTS gpkg not found: {args.nuts}", file=sys.stderr)
            return 1

    overview_path = None
    if args.overview:
        overview_path = args.overview_path
        if overview_path is None:
            overview_path = args.out_dir.parent / "lucas_greece_lc_lu_overview.png"

    run(
        args.lucas,
        args.nuts,
        args.out_dir,
        overview_path=overview_path,
        combo_mode=args.combo,
        top_n_combo=args.top,
        dpi=args.dpi,
        marker_size=args.size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
