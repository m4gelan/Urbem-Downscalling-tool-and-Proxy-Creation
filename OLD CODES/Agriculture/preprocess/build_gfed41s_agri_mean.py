"""
Build GFED4.1s agricultural fire climatology artefacts (run once).

Reads all GFED4.1s_<year>.hdf5 files (1997-2015) and produces:
  data/Agriculture/gfed41s_agri_dm_mean.npy  -- (720,1440) float32, kg DM m-2 yr-1
  data/Agriculture/gfed41s_grid_area.npy     -- (720,1440) float32, m2 per pixel
  data/Agriculture/gfed41s_lat.npy           -- (720,1440) float32, degrees_north
  data/Agriculture/gfed41s_lon.npy           -- (720,1440) float32, degrees_east
  data/Agriculture/gfed41s_nuts2_lookup.parquet
      columns: lat_idx (int), lon_idx (int), NUTS_ID (str), COUNTRY (str)
      One row per GFED pixel whose centre falls inside a NUTS-2 polygon and
      whose multi-year mean AGRI DM > 0.

Usage (from project root)::

    python -m Agriculture.preprocess.build_gfed41s_agri_mean

Override paths via environment variables or pass --gfed-dir / --nuts-gpkg / --out-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Defaults (resolved relative to project root = parent of Agriculture/)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PKG_ROOT = _HERE.parents[2]          # PDM_local/

_DEFAULT_GFED_DIR = _PKG_ROOT / "data" / "Agriculture" / "fire_emissions_v4_R1_1293" / "data"
_DEFAULT_NUTS_GPKG = _PKG_ROOT / "data" / "geometry" / "NUTS_RG_20M_2021_3035.gpkg"
_DEFAULT_OUT_DIR = _PKG_ROOT / "data" / "Agriculture"

# EU bounding box used to restrict the lookup to relevant pixels.
_EU_LAT = (34.0, 72.0)
_EU_LON = (-13.0, 46.0)


# ---------------------------------------------------------------------------
# Step 1: compute multi-year mean annual AGRI DM
# ---------------------------------------------------------------------------

def compute_agri_dm_mean(
    gfed_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns lat2d, lon2d, area_m2, dm_agri_mean  (all shape 720 x 1440, float32).

    dm_agri_mean is the 19-year (1997-2015) mean annual agricultural dry matter
    burned, in kg DM m-2 yr-1.  Actual agricultural DM per pixel = dm_agri_mean * area_m2.

    GFED4.1s structure per month:
      emissions/<MM>/DM            -- total DM, kg DM m-2 month-1
      emissions/<MM>/partitioning/DM_AGRI  -- AGRI fraction, unitless [0,1]
    """
    hdf5_files = sorted(gfed_dir.glob("GFED4.1s_*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(
            f"No GFED4.1s_*.hdf5 files found in {gfed_dir}. "
            "Download from https://globalfiredata.org/pages/data/"
        )

    print(f"  Found {len(hdf5_files)} GFED4.1s annual files: "
          f"{hdf5_files[0].stem} .. {hdf5_files[-1].stem}")

    accum: np.ndarray | None = None
    lat2d = lon2d = area = None

    for fp in hdf5_files:
        with h5py.File(fp, "r") as f:
            if accum is None:
                lat2d = f["lat"][:].astype(np.float32)
                lon2d = f["lon"][:].astype(np.float32)
                area  = f["ancill/grid_cell_area"][:].astype(np.float32)
                accum = np.zeros((720, 1440), dtype=np.float64)

            for m in range(1, 13):
                dm   = f[f"emissions/{m:02d}/DM"][:].astype(np.float64)
                frac = f[f"emissions/{m:02d}/partitioning/DM_AGRI"][:].astype(np.float64)
                # Guard against fill values / negatives
                valid = (dm > 0) & (frac >= 0) & (frac <= 1)
                accum += np.where(valid, dm * frac, 0.0)

    dm_mean = (accum / len(hdf5_files)).astype(np.float32)
    return lat2d, lon2d, area, dm_mean


# ---------------------------------------------------------------------------
# Step 2: build pixel -> NUTS-2 lookup
# ---------------------------------------------------------------------------

def build_nuts2_lookup(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    dm_mean: np.ndarray,
    nuts_gpkg: Path,
    eu_lat: tuple[float, float] = _EU_LAT,
    eu_lon: tuple[float, float] = _EU_LON,
) -> pd.DataFrame:
    """
    For each GFED pixel whose centre falls in the EU bounding box and whose
    multi-year mean AGRI DM > 0, find the NUTS-2 region containing that centre.

    Returns a DataFrame with columns:
        lat_idx  (int)  -- row index in the 720x1440 grid
        lon_idx  (int)  -- column index in the 720x1440 grid
        NUTS_ID  (str)
        COUNTRY  (str)  -- ISO country code (CNTR_CODE from NUTS GeoPackage)
    """
    print(f"  Loading NUTS-2 geometries from {nuts_gpkg} ...")
    nuts_gdf = gpd.read_file(nuts_gpkg)
    nuts2 = nuts_gdf[nuts_gdf["LEVL_CODE"] == 2].copy()
    if nuts2.crs is None or nuts2.crs.to_epsg() != 4326:
        nuts2 = nuts2.to_crs("EPSG:4326")
    nuts2 = nuts2[["NUTS_ID", "CNTR_CODE", "geometry"]].rename(
        columns={"CNTR_CODE": "COUNTRY"}
    )

    # Select EU pixels with positive AGRI DM
    eu_mask = (
        (lat2d >= eu_lat[0]) & (lat2d <= eu_lat[1]) &
        (lon2d >= eu_lon[0]) & (lon2d <= eu_lon[1]) &
        (dm_mean > 0)
    )
    rows_idx, cols_idx = np.where(eu_mask)
    n_pixels = len(rows_idx)
    print(f"  EU pixels with AGRI DM > 0: {n_pixels:,}")

    pts = gpd.GeoDataFrame(
        {"lat_idx": rows_idx.astype(np.int32),
         "lon_idx": cols_idx.astype(np.int32)},
        geometry=[
            Point(float(lon2d[r, c]), float(lat2d[r, c]))
            for r, c in zip(rows_idx, cols_idx)
        ],
        crs="EPSG:4326",
    )

    print("  Spatial join pixels -> NUTS-2 ...")
    joined = gpd.sjoin(pts, nuts2, how="left", predicate="within")
    joined = joined.dropna(subset=["NUTS_ID"]).copy()
    joined = joined[["lat_idx", "lon_idx", "NUTS_ID", "COUNTRY"]].reset_index(drop=True)
    joined["lat_idx"] = joined["lat_idx"].astype(np.int32)
    joined["lon_idx"] = joined["lon_idx"].astype(np.int32)
    print(f"  Pixels matched to a NUTS-2 region: {len(joined):,}")
    return joined


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    gfed_dir: Path = _DEFAULT_GFED_DIR,
    nuts_gpkg: Path = _DEFAULT_NUTS_GPKG,
    out_dir: Path = _DEFAULT_OUT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: computing 19-year mean AGRI DM from GFED4.1s ...")
    lat2d, lon2d, area, dm_mean = compute_agri_dm_mean(gfed_dir)

    np.save(out_dir / "gfed41s_agri_dm_mean.npy", dm_mean)
    np.save(out_dir / "gfed41s_grid_area.npy", area)
    np.save(out_dir / "gfed41s_lat.npy", lat2d)
    np.save(out_dir / "gfed41s_lon.npy", lon2d)
    print(f"  Saved: gfed41s_agri_dm_mean.npy  (max={dm_mean.max():.4e} kg DM/m2/yr)")
    print(f"  Saved: gfed41s_grid_area.npy, gfed41s_lat.npy, gfed41s_lon.npy")

    print("Step 2: building pixel -> NUTS-2 lookup ...")
    lookup = build_nuts2_lookup(lat2d, lon2d, dm_mean, nuts_gpkg)
    out_path = out_dir / "gfed41s_nuts2_lookup.parquet"
    lookup.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name}  ({len(lookup):,} rows)")

    print("Done.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--gfed-dir",  type=Path, default=_DEFAULT_GFED_DIR,
                   help="Directory containing GFED4.1s_*.hdf5 files.")
    p.add_argument("--nuts-gpkg", type=Path, default=_DEFAULT_NUTS_GPKG,
                   help="NUTS-2 GeoPackage (EPSG:4326 or reprojected automatically).")
    p.add_argument("--out-dir",   type=Path, default=_DEFAULT_OUT_DIR,
                   help="Directory to write output .npy and .parquet files.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(gfed_dir=args.gfed_dir, nuts_gpkg=args.nuts_gpkg, out_dir=args.out_dir)
