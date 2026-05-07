"""
Quick preview for rasters downloaded via WEkEO/HDA.

Usage (PowerShell):
  python Waste\\Auxiliaries\\preview_wekeo_rasters.py
  python Waste\\Auxiliaries\\preview_wekeo_rasters.py --file path\\to\\some.tif --out preview.png
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def _project_root() -> Path:
    # <root>/Waste/Auxiliaries/this_file.py
    return Path(__file__).resolve().parents[2]


RASTER_EXTS = {".tif", ".tiff", ".jp2", ".nc"}
ARCHIVE_EXTS = {".zip"}


def _find_candidates(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    candidates: list[Path] = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in RASTER_EXTS | ARCHIVE_EXTS:
            candidates.append(p)
    return candidates


def _first_raster_inside_zip(z: Path) -> str | None:
    try:
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist() if Path(n).suffix.lower() in RASTER_EXTS]
            return sorted(names)[0] if names else None
    except Exception:
        return None


def _plot_raster(path: Path, *, out: Path | None, verbose: bool = False) -> int:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
    except Exception as e:
        raise RuntimeError(
            "Missing packages for preview. Install: pip install rasterio matplotlib numpy"
        ) from e

    with rasterio.open(path) as ds:
        # Read band 1; downsample for very large rasters.
        overview = ds.overviews(1)
        if overview:
            factor = overview[-1]
            out_h = max(1, ds.height // factor)
            out_w = max(1, ds.width // factor)
            arr = ds.read(1, out_shape=(out_h, out_w))
        else:
            # Mild downsample if huge
            max_dim = 4000
            if max(ds.width, ds.height) > max_dim:
                scale = max(ds.width / max_dim, ds.height / max_dim)
                out_h = max(1, int(ds.height / scale))
                out_w = max(1, int(ds.width / scale))
                arr = ds.read(1, out_shape=(out_h, out_w))
            else:
                arr = ds.read(1)

        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        if verbose:
            print(f"Raster read: shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")

        plt.figure(figsize=(10, 8))
        plt.title(f"{path.name}\n{ds.crs}  {ds.width}x{ds.height}  dtype={arr.dtype}")
        plt.imshow(arr, cmap="viridis")
        plt.colorbar(label="Band 1 value")
        plt.axis("off")

        if out is None:
            plt.show()
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=160, bbox_inches="tight")
            print(f"Wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview a raster from data/Waste/wekeo_hrl_imp.")
    parser.add_argument(
        "--data-dir",
        default=str(_project_root() / "data" / "Waste" / "wekeo_hrl_imp"),
        help="Folder containing downloaded products.",
    )
    parser.add_argument("--file", type=str, default=None, help="Specific raster file to preview.")
    parser.add_argument("--out", type=str, default=None, help="Optional PNG output path.")
    parser.add_argument("--verbose", action="store_true", help="Print debug info (shapes, etc.).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out = Path(args.out).resolve() if args.out else None

    if args.file:
        path = Path(args.file).resolve()
        if not path.exists():
            raise FileNotFoundError(str(path))
        return _plot_raster(path, out=out, verbose=bool(args.verbose))

    candidates = _find_candidates(data_dir)
    if not candidates:
        print(f"No raster candidates found in {data_dir}")
        print("Expected files like .tif/.tiff/.jp2/.nc or .zip archives containing them.")
        return 2

    # Prefer direct rasters over archives.
    rasters = [p for p in candidates if p.suffix.lower() in RASTER_EXTS]
    if rasters:
        print(f"Previewing {rasters[0]}")
        return _plot_raster(rasters[0], out=out, verbose=bool(args.verbose))

    z = candidates[0]
    inner = _first_raster_inside_zip(z)
    if inner is None:
        print(f"Found archive but no rasters inside: {z}")
        return 2

    # Extract only one file to a temp-like location under the data dir.
    extracted = data_dir / "_preview_extract" / Path(inner).name
    extracted.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        with zf.open(inner) as src, open(extracted, "wb") as dst:
            dst.write(src.read())
    print(f"Extracted {inner} -> {extracted}")
    return _plot_raster(extracted, out=out, verbose=bool(args.verbose))


if __name__ == "__main__":
    raise SystemExit(main())

