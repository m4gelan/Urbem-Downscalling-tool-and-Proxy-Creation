#!/usr/bin/env python3
"""
Summarize and analyze UrbEm emissions script results.

Reads the CSVs and optional shapefiles from a Results folder (e.g.
Athens/Results_Folder/Emissions/2016/V3/Increase_Factor_1/Results),
prints totals by SNAP and pollutant, and optionally plots a choropleth map.

Usage:
    python code/scripts/analyze_urbem_emissions_results.py
    python code/scripts/analyze_urbem_emissions_results.py --results-dir "path/to/Results"
    python code/scripts/analyze_urbem_emissions_results.py --results-dir "path/to/Results" --map --snap 7 --pollutant NOx
"""

import argparse
from pathlib import Path

import pandas as pd


def find_csv(csv_dir: Path, pattern: str):
    """Return first CSV in csv_dir whose name contains pattern."""
    if not csv_dir.is_dir():
        return None
    for f in csv_dir.glob("*.csv"):
        if pattern in f.name:
            return f
    return None


def load_stat_csv(path: Path, sep: str = ";"):
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path, sep=sep, encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return None


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def analyze_total_stat(df: pd.DataFrame, label: str):
    """Print summary of a total_stat-style DataFrame (Snap/Region, pollutants, Count)."""
    if df is None or df.empty:
        print(f"{label}: no data.")
        return
    print_section(f"Total statistics ({label})")
    # Header row: first column is sector/snap name, rest are pollutants
    cols = list(df.columns)
    if len(cols) < 2:
        print(df.to_string())
        return
    # Show totals row (usually first) and optionally last rows
    print(df.head(1).to_string(index=False))
    print("  ...")
    print(f"  (rows: {len(df)})")
    # Sum by pollutant over all rows if numeric
    pollutant_cols = [c for c in cols[1:] if c != "Count" and df[c].dtype in ("float64", "int64")]
    if pollutant_cols:
        totals = df[pollutant_cols].sum()
        print("\n  Sum over all sectors (raw):")
        for c in pollutant_cols:
            print(f"    {c}: {totals[c]:.2e}")


def analyze_urbem_vs_cams(urbem_df: pd.DataFrame, cams_df: pd.DataFrame, label: str):
    """Compare UrbEm vs CAMS totals by SNAP."""
    if (urbem_df is None or urbem_df.empty) or (cams_df is None or cams_df.empty):
        print(f"UrbEm vs CAMS ({label}): missing data.")
        return
    print_section(f"UrbEm vs CAMS by SNAP ({label})")
    # Expect first column SNAP, then CH4, NOX, ...
    snap_col = urbem_df.columns[0]
    poll_cols = [c for c in urbem_df.columns[1:] if "(" in c or c in ("CH4", "NOx", "NOX", "NMVOC", "CO", "SO2", "NH3", "PM25", "PM10")]
    if not poll_cols:
        poll_cols = list(urbem_df.columns[1:])
    merged = urbem_df.merge(cams_df, on=snap_col, suffixes=("_urbem", "_cams"), how="outer")
    # Simplify: show one pollutant (e.g. NOx) comparison
    nox_urbem = [c for c in merged.columns if "NOX" in c.upper() and "urbem" in c]
    nox_cams = [c for c in merged.columns if "NOX" in c.upper() and "cams" in c]
    if nox_urbem and nox_cams:
        merged_sub = merged[[snap_col, nox_urbem[0], nox_cams[0]]].copy()
        merged_sub.columns = [snap_col, "NOx_urbem", "NOx_cams"]
        merged_sub["ratio"] = merged_sub["NOx_cams"] / merged_sub["NOx_urbem"].replace(0, float("nan"))
        print(merged_sub.to_string(index=False))
    else:
        print(merged.head(15).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Analyze UrbEm emissions results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Athens/Results_Folder/Emissions/2016/V3/Increase_Factor_1/Results"),
        help="Path to Results folder (contains Results_CSVs/ and .shp files)",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="Plot a choropleth map for one SNAP and pollutant (requires geopandas, matplotlib)",
    )
    parser.add_argument("--snap", type=int, default=7, help="SNAP sector for map (default 7)")
    parser.add_argument("--pollutant", type=str, default="NOx", help="Pollutant column for map (e.g. NOx, PM10)")
    parser.add_argument("--no-open", action="store_true", help="Do not open map in browser / show window")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    csv_dir = results_dir / "Results_CSVs"
    if not csv_dir.is_dir():
        print(f"Results_CSVs not found under {results_dir}")
        return 1

    # Discover CSVs
    total_areas = find_csv(csv_dir, "total_stat_areas")
    total_lines = find_csv(csv_dir, "total_stat_lines")
    urbem_areas = find_csv(csv_dir, "urbem_stat_areas")
    cams_areas = find_csv(csv_dir, "cams_stat_areas")
    urbem_lines = find_csv(csv_dir, "urbem_stat_lines")
    cams_lines = find_csv(csv_dir, "cams_stat_lines")

    print(f"Results dir: {results_dir}")
    print(f"CSV folder:  {csv_dir}")

    # Load and print total stats (areas)
    df_total_areas = load_stat_csv(total_areas)
    analyze_total_stat(df_total_areas, "areas")

    # Load and print total stats (lines)
    df_total_lines = load_stat_csv(total_lines)
    analyze_total_stat(df_total_lines, "lines")

    # UrbEm vs CAMS
    df_urbem_a = load_stat_csv(urbem_areas)
    df_cams_a = load_stat_csv(cams_areas)
    analyze_urbem_vs_cams(df_urbem_a, df_cams_a, "areas")

    if urbem_lines and cams_lines:
        df_urbem_l = load_stat_csv(urbem_lines)
        df_cams_l = load_stat_csv(cams_lines)
        analyze_urbem_vs_cams(df_urbem_l, df_cams_l, "lines")

    # Optional map
    if args.map:
        shp_path = results_dir / f"CAMS_emissions_final_snap_{args.snap}.shp"
        if not shp_path.exists():
            shp_path = results_dir / f"urbem_final_snap_{args.snap}.shp"
        if not shp_path.exists():
            print(f"\nNo shapefile found for SNAP {args.snap} (tried CAMS_emissions_final_snap_*.shp, urbem_final_snap_*.shp)")
        else:
            try:
                import geopandas as gpd
                import matplotlib.pyplot as plt
                gdf = gpd.read_file(shp_path)
                col = args.pollutant
                if col not in gdf.columns:
                    alt = [c for c in gdf.columns if col.upper() in c.upper() or "NOX" in c.upper() and col.upper() == "NOX"]
                    col = alt[0] if alt else gdf.columns[-1]
                gdf = gdf.to_crs("EPSG:4326")
                fig, ax = plt.subplots(1, 1, figsize=(10, 8))
                gdf.plot(column=col, ax=ax, legend=True, cmap="YlOrRd", scheme="quantiles", k=7)
                ax.set_title(f"SNAP {args.snap} - {col}")
                ax.set_axis_off()
                out_map = results_dir / f"map_snap_{args.snap}_{col}.png"
                plt.savefig(out_map, dpi=150, bbox_inches="tight")
                print(f"\nMap saved: {out_map}")
                if not args.no_open:
                    plt.show()
            except ImportError as e:
                print(f"\nMap skipped (install geopandas, matplotlib): {e}")
            except Exception as e:
                print(f"\nMap error: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
