"""
Quick view of the merged Greece imperviousness mosaic (or any GeoTIFF).

Usage:
  python Waste\\Auxiliaries\\preview_wekeo_mosaic.py
  python Waste\\Auxiliaries\\preview_wekeo_mosaic.py --out preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_MOSAIC = _project_root() / "data" / "Waste" / "wekeo_hrl_imp" / "greece_imp_100m_mosaic.tif"


def main() -> int:
    p = argparse.ArgumentParser(description="Preview WEkEO HRL merged mosaic GeoTIFF.")
    p.add_argument(
        "--tif",
        type=str,
        default=str(DEFAULT_MOSAIC),
        help="Path to GeoTIFF (default: greece_imp_100m_mosaic.tif).",
    )
    p.add_argument("--out", type=str, default=None, help="Save PNG instead of opening a window.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    path = Path(args.tif).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
    except ImportError as e:
        raise RuntimeError("Install: pip install rasterio matplotlib numpy") from e

    with rasterio.open(path) as ds:
        overview = ds.overviews(1)
        if overview:
            factor = overview[-1]
            out_h = max(1, ds.height // factor)
            out_w = max(1, ds.width // factor)
            arr = ds.read(1, out_shape=(out_h, out_w))
        else:
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

        if args.verbose:
            print(f"shape={arr.shape} dtype={arr.dtype} crs={ds.crs}")

        plt.figure(figsize=(11, 9))
        plt.title(f"{path.name}\n{ds.crs}  {ds.width}x{ds.height}")
        plt.imshow(arr, cmap="viridis")
        plt.colorbar(label="Band 1")
        plt.axis("off")

        if args.out:
            out = Path(args.out).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=160, bbox_inches="tight")
            print(f"Wrote {out}")
        else:
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
