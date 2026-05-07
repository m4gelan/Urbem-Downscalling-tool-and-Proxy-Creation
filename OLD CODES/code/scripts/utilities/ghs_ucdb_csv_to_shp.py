#!/usr/bin/env python3
"""
Convert GHS Urban Centre Database (CSV with bounding boxes) to shapefile.

Reads the full CSV from data/GHS_STAT_UCDB2015MT_GLOBE_R2019A/ and builds
polygons from columns BBX_LATMN, BBX_LONMN, BBX_LATMX, BBX_LONMX, then
writes a shapefile in the same folder. Output name matches what UrbEm expects:
GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_0.shp (in a subfolder or same folder).

Usage:
    python code/scripts/utilities/ghs_ucdb_csv_to_shp.py
    python code/scripts/utilities/ghs_ucdb_csv_to_shp.py --input-dir "path/to/GHS_STAT_..." --output "path/to/output.shp"
"""

import argparse
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

WGS84 = "EPSG:4326"


def bbox_to_polygon(row):
    """Build a Shapely polygon from bounding box columns (lat/lon min/max)."""
    lat_min = row["BBX_LATMN"]
    lon_min = row["BBX_LONMN"]
    lat_max = row["BBX_LATMX"]
    lon_max = row["BBX_LONMX"]
    if pd.isna(lat_min) or pd.isna(lon_min) or pd.isna(lat_max) or pd.isna(lon_max):
        return None
    return Polygon(
        [
            (float(lon_min), float(lat_min)),
            (float(lon_max), float(lat_min)),
            (float(lon_max), float(lat_max)),
            (float(lon_min), float(lat_max)),
            (float(lon_min), float(lat_min)),
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert GHS-UCDB CSV (bbox) to shapefile."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "data" / "GHS_STAT_UCDB2015MT_GLOBE_R2019A",
        help="Directory containing the GHS UCDB CSV file(s)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.csv",
        help="CSV filename (default: full table with bbox columns)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (.shp). Default: input_dir/../GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_0/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_0.shp (for UrbEm InFolder)",
    )
    parser.add_argument(
        "--keep-columns",
        type=str,
        nargs="*",
        default=["ID_HDC_G0", "AREA", "CTR_MN_ISO", "UC_NM_MN", "GCPNT_LAT", "GCPNT_LON"],
        help="Columns to keep in the shapefile (default: ID_HDC_G0, AREA, CTR_MN_ISO, UC_NM_MN, GCPNT_LAT, GCPNT_LON)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    csv_path = input_dir / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_path = args.output
    if out_path is None:
        # UrbEm expects: InFolder/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_0/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_0.shp
        out_dir = input_dir.parent / "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_0"
        out_path = out_dir / "GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_0.shp"

    print(f"Reading {csv_path}...")
    encodings = ["utf-8", "cp1252", "latin-1", "iso-8859-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, low_memory=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError(f"Could not decode CSV with any of: {', '.join(encodings)}")

    required = ["BBX_LATMN", "BBX_LONMN", "BBX_LATMX", "BBX_LONMX"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    print("Building polygons from bounding boxes...")
    df["geometry"] = df.apply(bbox_to_polygon, axis=1)
    df = df.dropna(subset=["geometry"])

    keep = [c for c in args.keep_columns if c in df.columns]
    df = df[keep + ["geometry"]].copy()

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=WGS84)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {out_path}...")
    gdf.to_file(out_path, driver="ESRI Shapefile")
    print(f"Done. {len(gdf)} features written.")


if __name__ == "__main__":
    main()
