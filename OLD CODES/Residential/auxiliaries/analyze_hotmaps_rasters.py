#!/usr/bin/env python3
"""
Compare Hotmaps GeoTIFFs (res vs non-res heat and GFA): paths, shape, CRS, stats, file identity.

Usage (from project root)::

  python Residential/auxiliaries/analyze_hotmaps_rasters.py
  python Residential/auxiliaries/analyze_hotmaps_rasters.py --config path/to/residential_cams_downscale.config.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(project_root: Path, p: str | Path) -> Path:
    x = Path(p)
    return x if x.is_absolute() else (project_root / x)


def _file_digest(path: Path, max_bytes: int = 8_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        n = 0
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            if n >= max_bytes:
                h.update(str(path.stat().st_size).encode())
                break
    return h.hexdigest()[:16]


def _summarize_raster(path: Path, *, preview: int = 4096) -> dict:
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    out: dict = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return out
    out["size_bytes"] = path.stat().st_size
    out["sha256_prefix"] = _file_digest(path)
    with rasterio.open(path) as src:
        out["crs"] = str(src.crs) if src.crs else None
        out["shape"] = (int(src.height), int(src.width))
        out["count"] = int(src.count)
        out["dtype"] = str(src.dtypes[0])
        out["nodata"] = src.nodata
        out["bounds"] = [float(x) for x in src.bounds]
        pw = min(int(src.width), preview)
        ph = min(int(src.height), preview)
        h, w = int(src.height), int(src.width)
        row_off = max(0, (h - ph) // 2)
        col_off = max(0, (w - pw) // 2)
        win = Window(col_off, row_off, pw, ph)
        arr = src.read(1, window=win, masked=True)
        out["stats_window"] = (
            f"center {col_off},{row_off} {pw}x{ph} (preview; full {w}x{h})"
        )
        data = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(data)
        if hasattr(arr, "mask"):
            valid = valid & ~np.asarray(arr.mask)
        out["valid_fraction_in_window"] = float(np.mean(valid)) if valid.size else 0.0
        if np.any(valid):
            v = data[valid]
            out["min_in_window"] = float(np.min(v))
            out["max_in_window"] = float(np.max(v))
            out["mean_in_window"] = float(np.mean(v))
        else:
            out["min_in_window"] = out["max_in_window"] = out["mean_in_window"] = None
    return out


def _compare_pair(
    name_a: str,
    path_a: Path,
    name_b: str,
    path_b: Path,
    *,
    subsample: int,
    preview: int = 4096,
) -> None:
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    print(f"\n--- Pair: {name_a} vs {name_b} ---")
    if not path_a.is_file() or not path_b.is_file():
        print("  skip (missing file)")
        return
    if path_a.resolve() == path_b.resolve():
        print("  SAME FILE PATH (hard-linked or identical path string).")
    sa = path_a.stat().st_size
    sb = path_b.stat().st_size
    if sa == sb:
        print(f"  same byte size ({sa} B) — possible duplicate content")
    da = _file_digest(path_a)
    db = _file_digest(path_b)
    if da == db:
        print(f"  same partial-file hash prefix ({da}) — very likely identical bytes")
    else:
        print(f"  different partial-file hash: {da} vs {db}")

    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        if a.shape != b.shape:
            print(f"  different shape: {a.shape} vs {b.shape}")
            return
        if str(a.crs) != str(b.crs):
            print(f"  CRS differ: {a.crs} vs {b.crs}")
        ta = np.allclose(np.array(a.transform), np.array(b.transform), rtol=0, atol=0)
        if not ta:
            print(f"  transforms differ:\n    {a.transform}\n    {b.transform}")

        h, w = int(a.height), int(a.width)
        pw = min(w, preview)
        ph = min(h, preview)
        row_off = max(0, (h - ph) // 2)
        col_off = max(0, (w - pw) // 2)
        win = Window(col_off, row_off, pw, ph)
        print(f"  comparison window: center {pw}x{ph} at col={col_off} row={row_off}")
        a1 = a.read(1, window=win, masked=True)
        b1 = b.read(1, window=win, masked=True)
        x = np.asarray(a1, dtype=np.float64)
        y = np.asarray(b1, dtype=np.float64)
        ma = np.asarray(a1.mask) if hasattr(a1, "mask") else np.zeros_like(x, dtype=bool)
        mb = np.asarray(b1.mask) if hasattr(b1, "mask") else np.zeros_like(y, dtype=bool)
        valid = np.isfinite(x) & np.isfinite(y) & ~ma & ~mb
        n = int(np.count_nonzero(valid))
        if n == 0:
            print("  no overlapping valid pixels for diff")
            return
        if subsample > 0 and n > subsample:
            rng = np.random.default_rng(0)
            idx = np.flatnonzero(valid)
            pick = rng.choice(idx, size=subsample, replace=False)
            valid_flat = np.zeros_like(valid.ravel(), dtype=bool)
            valid_flat[pick] = True
            valid = valid_flat.reshape(valid.shape)
        d = np.abs(x[valid] - y[valid])
        print(
            f"  overlapping valid pixels (evaluated): {int(np.count_nonzero(valid))} "
            f"of {valid.size} ({100.0 * np.count_nonzero(valid) / valid.size:.2f}%)"
        )
        print(f"  max |a-b|: {float(np.max(d)):.6g}")
        print(f"  mean |a-b|: {float(np.mean(d)):.6g}")
        if np.allclose(x[valid], y[valid], rtol=1e-6, atol=1e-8):
            print("  arrays are allclose (rtol=1e-6, atol=1e-8) on evaluated pixels")
        else:
            print("  arrays are NOT allclose; datasets differ spatially")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Hotmaps rasters from residential config.")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="residential_cams_downscale.config.json",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: repo root)",
    )
    ap.add_argument(
        "--subsample",
        type=int,
        default=2_000_000,
        help="Max pixels for pair diff (0 = all)",
    )
    args = ap.parse_args()

    project_root = args.root or _root()
    cfg_path = args.config or (project_root / "Residential" / "config" / "residential_cams_downscale.config.json")
    cfg_path = cfg_path if cfg_path.is_absolute() else (project_root / cfg_path)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    hm = (cfg.get("paths") or {}).get("hotmaps") or {}
    keys = ("heat_res", "heat_nonres", "gfa_res", "gfa_nonres")
    paths: dict[str, Path] = {}
    for k in keys:
        rel = hm.get(k)
        paths[k] = _resolve(project_root, rel) if rel else Path()

    print("Hotmaps paths (from config)")
    for k in keys:
        p = paths[k]
        print(f"  {k}: {p}  [{'OK' if p.is_file() else 'MISSING'}]")

    print("\nPer-raster summary")
    for k in keys:
        p = paths[k]
        s = _summarize_raster(p)
        print(f"\n{k}:")
        for kk, vv in s.items():
            print(f"  {kk}: {vv}")

    sub = int(args.subsample)
    _compare_pair("heat_res", paths["heat_res"], "heat_nonres", paths["heat_nonres"], subsample=sub)
    _compare_pair("gfa_res", paths["gfa_res"], "gfa_nonres", paths["gfa_nonres"], subsample=sub)

    print(
        "\nNote: Identical grids (extent, resolution, transform) are normal for Hotmaps products; "
        "res vs non-res should still differ in values at each pixel. If max|a-b| is 0, "
        "the files are duplicates or the same dataset was saved twice."
    )


if __name__ == "__main__":
    main()
