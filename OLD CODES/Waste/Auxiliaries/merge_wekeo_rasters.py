"""
Merge downloaded WEkEO GeoTIFF tiles into one mosaic (same CRS).

HDA returns one file per tile; this script builds a single GeoTIFF for your bbox.

Usage:
  python Waste\\Auxiliaries\\merge_wekeo_rasters.py --glob "data/Waste/wekeo_hrl_imp/*.tif"
  python Waste\\Auxiliaries\\merge_wekeo_rasters.py --glob "data/Waste/wekeo_hrl_imp/**/*.tif" --out merged.tif
"""

from __future__ import annotations

import argparse
import glob as glob_mod
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Merge GeoTIFF tiles with rasterio.merge.")
    p.add_argument("--glob", dest="glob_pattern", type=str, required=True, help="Glob for .tif files.")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output GeoTIFF path (default: <first_dir>/mosaic_merged.tif).",
    )
    p.add_argument("--nodata", type=float, default=None, help="Optional nodata value for output.")
    args = p.parse_args()

    paths = sorted(glob_mod.glob(args.glob_pattern, recursive=True))
    paths = [Path(x) for x in paths if x.lower().endswith((".tif", ".tiff"))]
    if not paths:
        raise SystemExit(f"No .tif/.tiff files matched: {args.glob_pattern!r}")

    try:
        import rasterio
        from rasterio.merge import merge
    except ImportError as e:
        raise SystemExit("Install: pip install rasterio numpy") from e

    srcs = [rasterio.open(p) for p in paths]
    try:
        mosaic, out_transform = merge(srcs, nodata=args.nodata)
        out_meta = srcs[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "count": mosaic.shape[0],
            }
        )
        if args.nodata is not None:
            out_meta["nodata"] = args.nodata
    finally:
        for s in srcs:
            s.close()

    out_path = Path(args.out) if args.out else paths[0].parent / "mosaic_merged.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic)

    print(f"Wrote {out_path} ({mosaic.shape[2]}x{mosaic.shape[1]} px, {mosaic.shape[0]} bands)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
