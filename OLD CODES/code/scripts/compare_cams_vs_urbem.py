#!/usr/bin/env python3
"""
Side-by-side comparison map: original CAMS coarse grid vs UrbEm 1 km downscaled grid.

Both maps show emission density [arbitrary CAMS units / km²] for the same domain,
using the same log color scale so spatial redistribution is immediately visible.

Usage (from project root):
    python code/scripts/compare_cams_vs_urbem.py
    python code/scripts/compare_cams_vs_urbem.py --snap 7 --pollutant NOX
    python code/scripts/compare_cams_vs_urbem.py --snap 1 --pollutant PM10
    python code/scripts/compare_cams_vs_urbem.py --all-snaps --pollutant NOX

Available SNAPs  : 1 2 3 4 5 6 7 8 9 10 11 12
Available polls  : CH4 CO NH3 NMVOC NOX PM10 SO2  (PM25 for CAMS, PM2_5 for UrbEm handled auto)
"""

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

SNAP_LABELS = {
    1:  "Public Power",
    2:  "Stationary Combustion",
    3:  "Industry (SNAP 3)",
    4:  "Industry (SNAP 4)",
    5:  "Fugitives",
    6:  "Solvents",
    7:  "Road Transport",
    8:  "Shipping",
    9:  "Waste",
    10: "Agriculture",
    11: "Aviation",
    12: "OffRoad",
}

POLL_UNITS = {
    "NOX": "NOx",
    "PM10": "PM10",
    "PM2_5": "PM2.5",
    "SO2": "SO2",
    "NH3": "NH3",
    "NMVOC": "NMVOC",
    "CO": "CO",
    "CH4": "CH4",
}


def col_for_poll(gdf: gpd.GeoDataFrame, poll: str) -> str:
    """
    Return the _km density column for the requested pollutant.
    Both CAMS and UrbEm shapefiles store per-km² emission density in these columns
    (CAMS: NOX_km = NOX / 1000 / area_km2; UrbEm: NOX_km = distributed per 1km cell).
    Using _km for both gives directly comparable density values.
    PM2.5 naming differs: CAMS uses PM25_km, UrbEm uses PM2_5_km.
    """
    # Prefer the _km density column
    km_candidates = [
        poll + "_km",
        poll.upper() + "_km",
        # PM2.5 variants
        "PM25_km" if poll.upper() in ("PM2_5", "PM25", "PM2.5") else None,
        "PM2_5_km" if poll.upper() in ("PM2_5", "PM25", "PM2.5") else None,
    ]
    for candidate in km_candidates:
        if candidate and candidate in gdf.columns:
            return candidate
    raise KeyError(
        f"No _km density column found for '{poll}'. "
        f"Available _km columns: {[c for c in gdf.columns if c.endswith('_km')]}"
    )


def emission_density(gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    """
    Copy the pre-computed per-km² density column into a 'density' column.
    The _km columns in both CAMS and UrbEm shapefiles already represent
    emission intensity per km², so no area division is needed.
    """
    out = gdf.copy()
    out["density"] = out[col]
    return out


def make_norm(cams_gdf: gpd.GeoDataFrame, urb_gdf: gpd.GeoDataFrame) -> mcolors.Normalize:
    """Build a shared log normalisation from the combined non-zero density values."""
    combined = np.concatenate([
        cams_gdf["density"].values,
        urb_gdf["density"].values,
    ])
    combined = combined[combined > 0]
    if len(combined) == 0:
        return mcolors.LogNorm(vmin=1, vmax=10)
    vmin = np.percentile(combined, 5)
    vmax = np.percentile(combined, 99)
    # Ensure vmin < vmax and both positive
    vmin = max(vmin, combined.min())
    if vmin <= 0:
        vmin = combined[combined > 0].min()
    vmax = max(vmax, vmin * 10)
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def plot_comparison(
    cams_path: Path,
    urb_path: Path,
    poll: str,
    snap: int,
    out_path: Path,
    cmap: str = "YlOrRd",
    show: bool = True,
):
    cams_gdf = gpd.read_file(cams_path)
    urb_gdf  = gpd.read_file(urb_path)

    # Determine actual column names (PM25 vs PM2_5 etc.)
    cams_col = col_for_poll(cams_gdf, poll)
    urb_col  = col_for_poll(urb_gdf,  poll)

    # Replace any negative sentinel values (-999) with 0
    cams_gdf[cams_col] = cams_gdf[cams_col].clip(lower=0)
    urb_gdf[urb_col]   = urb_gdf[urb_col].clip(lower=0)

    # Compute density per km²
    cams_d = emission_density(cams_gdf, cams_col)
    urb_d  = emission_density(urb_gdf,  urb_col)

    # Shared log normalisation
    norm = make_norm(cams_d, urb_d)

    poll_label = POLL_UNITS.get(poll, poll)
    snap_label = SNAP_LABELS.get(snap, f"SNAP {snap}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: CAMS coarse grid ────────────────────────────────────────────────
    ax_cams = axes[0]
    cams_d.plot(
        column="density",
        ax=ax_cams,
        norm=norm,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.3,
        legend=False,
        missing_kwds={"color": "lightgrey"},
    )
    ax_cams.set_title(
        f"CAMS original  (coarse ~10 km)\n{snap_label}  ·  {poll_label}",
        fontsize=11, fontweight="bold",
    )
    ax_cams.set_xlabel("Easting (m, UTM 34N)")
    ax_cams.set_ylabel("Northing (m, UTM 34N)")
    ax_cams.tick_params(axis="both", labelsize=8)

    # ── Right: UrbEm downscaled 1 km grid ────────────────────────────────────
    ax_urb = axes[1]
    urb_d.plot(
        column="density",
        ax=ax_urb,
        norm=norm,
        cmap=cmap,
        edgecolor="none",
        legend=False,
        missing_kwds={"color": "lightgrey"},
    )
    ax_urb.set_title(
        f"UrbEm downscaled  (1 km)\n{snap_label}  ·  {poll_label}",
        fontsize=11, fontweight="bold",
    )
    ax_urb.set_xlabel("Easting (m, UTM 34N)")
    ax_urb.set_ylabel("")
    ax_urb.tick_params(axis="both", labelsize=8)

    # ── Same spatial extent for both ─────────────────────────────────────────
    bounds = urb_d.total_bounds  # [xmin, ymin, xmax, ymax]
    pad_x = (bounds[2] - bounds[0]) * 0.02
    pad_y = (bounds[3] - bounds[1]) * 0.02
    for ax in axes:
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    # ── Shared colorbar ───────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.02)
    cbar.set_label(f"{poll_label}  density  [CAMS units / km²]  (log scale)", fontsize=9)

    fig.suptitle(
        f"Emission downscaling comparison  ·  Athens 2016  ·  SNAP {snap}: {snap_label}",
        fontsize=13, fontweight="bold",
    )
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout(rect=[0, 0, 0.93, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare CAMS coarse vs UrbEm downscaled emissions.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Athens/Results_Folder/Emissions/2016/V3/Increase_Factor_1/Results"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to save PNGs (default: <results-dir>/Maps)",
    )
    parser.add_argument("--snap",       type=int,   default=7,     help="SNAP sector (default 7)")
    parser.add_argument("--pollutant",  type=str,   default="NOX", help="Pollutant column (default NOX)")
    parser.add_argument("--all-snaps",  action="store_true",       help="Loop over all available SNAPs")
    parser.add_argument("--cmap",       type=str,   default="YlOrRd")
    parser.add_argument("--no-show",    action="store_true",       help="Do not open interactive window")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    out_dir = (args.out_dir or results_dir / "Maps").resolve()

    snaps_to_run = list(SNAP_LABELS.keys()) if args.all_snaps else [args.snap]

    for snap in snaps_to_run:
        cams_path = results_dir / f"CAMS_emissions_final_snap_{snap}.shp"
        urb_path  = results_dir / f"urbem_final_snap_{snap}.shp"

        if not cams_path.exists() or not urb_path.exists():
            print(f"SNAP {snap}: shapefile(s) not found, skipping.")
            continue

        poll = args.pollutant.upper()
        out_path = out_dir / f"compare_snap{snap}_{poll}.png"

        try:
            plot_comparison(
                cams_path=cams_path,
                urb_path=urb_path,
                poll=poll,
                snap=snap,
                out_path=out_path,
                cmap=args.cmap,
                show=not args.no_show,
            )
        except Exception as e:
            print(f"SNAP {snap}: error - {e}")


if __name__ == "__main__":
    raise SystemExit(main())
