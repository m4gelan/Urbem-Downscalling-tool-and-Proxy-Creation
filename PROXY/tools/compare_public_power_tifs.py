#!/usr/bin/env python3
"""Compare legacy vs new Public Power area weight GeoTIFFs (same grid: coverage + value agreement)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    root = _root()
    ap = argparse.ArgumentParser(
        description="Compare SourceProxies legacy PublicPower_sourcearea.tif to PROXY output."
    )
    ap.add_argument(
        "--legacy",
        type=Path,
        default=root / "SourceProxies" / "outputs" / "EL" / "PublicPower_sourcearea.tif",
        help="Legacy GeoTIFF path.",
    )
    ap.add_argument(
        "--new",
        type=Path,
        default=root / "OUTPUT" / "Proxy_weights" / "A_PublicPower" / "publicpower_areasource.tif",
        help="New pipeline GeoTIFF path.",
    )
    args = ap.parse_args()
    legacy_p = args.legacy if args.legacy.is_absolute() else root / args.legacy
    new_p = args.new if args.new.is_absolute() else root / args.new

    if not legacy_p.is_file():
        print(f"Missing legacy file: {legacy_p}", file=sys.stderr)
        return 1
    if not new_p.is_file():
        print(f"Missing new file: {new_p}", file=sys.stderr)
        return 1

    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(legacy_p) as a, rasterio.open(new_p) as b:
        print("=== Raster metadata ===")
        print(f"Legacy: {legacy_p}")
        print(f"  shape={a.height}x{a.width} crs={a.crs} dtype={a.dtypes[0]} nodata={a.nodata}")
        print(f"  bounds={tuple(round(x, 2) for x in a.bounds)}")
        print(f"New:    {new_p}")
        print(f"  shape={b.height}x{b.width} crs={b.crs} dtype={b.dtypes[0]} nodata={b.nodata}")
        print(f"  bounds={tuple(round(x, 2) for x in b.bounds)}")

        raw_a = a.read(1).astype(np.float64)
        raw_b = b.read(1).astype(np.float64)

        nd_a, nd_b = a.nodata, b.nodata
        if nd_a is not None:
            raw_a = np.where(raw_a == float(nd_a), np.nan, raw_a)
        if nd_b is not None:
            raw_b = np.where(raw_b == float(nd_b), np.nan, raw_b)

        same_grid = (
            a.shape == b.shape
            and a.crs == b.crs
            and np.allclose(np.asarray(a.transform)[:6], np.asarray(b.transform)[:6], rtol=0, atol=1e-6)
        )

        if not same_grid:
            print("\n=== Different grid: reproject NEW onto LEGACY (bilinear) ===")
            arr_b = np.full_like(raw_a, np.nan, dtype=np.float64)
            reproject(
                source=raw_b,
                destination=arr_b,
                src_transform=b.transform,
                src_crs=b.crs,
                dst_transform=a.transform,
                dst_crs=a.crs,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            raw_b = arr_b

        pos_a = np.isfinite(raw_a) & (raw_a > 0)
        pos_b = np.isfinite(raw_b) & (raw_b > 0)
        zero_a = np.isfinite(raw_a) & (raw_a == 0)
        zero_b = np.isfinite(raw_b) & (raw_b == 0)

        print("\n=== Nodata / zero (read carefully) ===")
        print(
            "Legacy nodata=None: zeros in the file are real numeric zeros (eligible pixels with 0 share)."
        )
        print(
            "New nodata=0.0: writer marks 0 as nodata; for *coverage* we still count raw values > 0 below."
        )

        print("\n=== Coverage (strictly positive weight share) ===")
        na = int(np.sum(pos_a))
        nb = int(np.sum(pos_b))
        both = int(np.sum(pos_a & pos_b))
        only_leg = int(np.sum(pos_a & ~pos_b))
        only_new = int(np.sum(pos_b & ~pos_a))
        print(f"  legacy > 0:     {na:>10}  ({100.0 * na / raw_a.size:.2f}% of grid)")
        print(f"  new > 0:        {nb:>10}  ({100.0 * nb / raw_b.size:.2f}% of grid)")
        print(f"  both > 0:       {both:>10}")
        print(f"  legacy>0 only: {only_leg:>10}  (mass in legacy missing in new)")
        print(f"  new>0 only:     {only_new:>10}  (mass in new not in legacy)")
        print(f"  both == 0:      {int(np.sum(zero_a & zero_b)):>10}")

        print("\n=== Value agreement where BOTH allocate (legacy>0 and new>0) ===")
        m_pos = pos_a & pos_b
        if not np.any(m_pos):
            print("  No overlapping positive pixels.")
        else:
            aa = raw_a[m_pos]
            bb = raw_b[m_pos]
            d = bb - aa
            print(f"  pixels: {int(np.sum(m_pos))}")
            print(
                f"  new - legacy: min={np.min(d):.6g} max={np.max(d):.6g} "
                f"mean={np.mean(d):.6g} mae={np.mean(np.abs(d)):.6g} "
                f"rmse={float(np.sqrt(np.mean(d * d))):.6g}"
            )
            if np.std(aa) > 0 and np.std(bb) > 0:
                print(f"  Pearson r: {float(np.corrcoef(aa, bb)[0, 1]):.6f}")
            denom = np.maximum(np.abs(aa) + np.abs(bb), 1e-15)
            sym_rel = np.abs(d) / denom
            print(
                f"  |new-leg|/(|leg|+|new|): median={float(np.median(sym_rel)):.6g} "
                f"p95={float(np.percentile(sym_rel, 95)):.6g}"
            )

        print("\n=== Global sums (finite pixels only) ===")
        fa = np.isfinite(raw_a)
        fb = np.isfinite(raw_b)
        print(f"  sum(legacy) all finite: {float(np.nansum(raw_a[fa])):.6g}")
        print(f"  sum(new) all finite:    {float(np.nansum(raw_b[fb])):.6g}")
        print(f"  sum(legacy) pos only:   {float(np.nansum(raw_a[pos_a])):.6g}")
        print(f"  sum(new) pos only:      {float(np.nansum(raw_b[pos_b])):.6g}")

        print("\n=== Interpretation ===")
        if nb < 0.5 * na:
            print(
                "  New raster has far fewer positive pixels than legacy: different CAMS cell set, "
                "bbox filter, CORINE/pop paths, or builder logic — not just a viz scaling issue."
            )
        if both < 0.1 * min(na, nb):
            print(
                "  Very little spatial overlap of positive mass: pipelines place weight on "
                "different footprints even on the same grid."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
