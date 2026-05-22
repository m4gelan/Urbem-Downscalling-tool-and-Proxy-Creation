#!/usr/bin/env python3
"""Merge Copernicus imperviousness tile zips into one GeoTIFF."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path

import rasterio
from rasterio.merge import merge

# After downloading the imperviousness raster from COPERNICS you get a zip bundle.
# Change INPUT_DIR to that zip (or a folder with Results/*.zip) and OUTPUT_RASTER.
INPUT_DIR = "INPUT/78455.zip"
OUTPUT_RASTER = "INPUT/Proxy/ProxySpecific/Waste/imperviousness_density_2021_AUS.tif"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_under_root(root: Path, p: str | Path) -> Path:
    q = Path(p)
    return q.resolve() if q.is_absolute() else (root / q).resolve()


def extract_tile_tifs(source: Path, work: Path) -> list[Path]:
    tifs: list[Path] = []

    def from_inner_zip(inner_zip: Path) -> None:
        with zipfile.ZipFile(inner_zip) as zf:
            for name in zf.namelist():
                if name.lower().endswith(".tif"):
                    out = work / Path(name).name
                    out.write_bytes(zf.read(name))
                    tifs.append(out)

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as outer:
            for name in outer.namelist():
                if not name.lower().endswith(".zip"):
                    continue
                inner_path = work / Path(name).name
                inner_path.write_bytes(outer.read(name))
                from_inner_zip(inner_path)
        return tifs

    results = source / "Results" if (source / "Results").is_dir() else source
    for inner_zip in sorted(results.glob("*.zip")):
        from_inner_zip(inner_zip)
    if not tifs:
        raise SystemExit(f"No tile zips found under {source}")
    return tifs


def main() -> None:
    root = project_root()
    source = resolve_under_root(root, INPUT_DIR)
    out_path = resolve_under_root(root, OUTPUT_RASTER)
    if not source.exists():
        raise SystemExit(f"Input not found: {source}")

    work = Path(tempfile.mkdtemp(prefix="imperviousness_"))
    try:
        tifs = extract_tile_tifs(source, work)
        print(f"Found {len(tifs)} tiles")

        srcs = [rasterio.open(p) for p in tifs]
        try:
            mosaic, transform = merge(srcs)
            profile = srcs[0].profile.copy()
            profile.update(
                driver="GTiff",
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                count=mosaic.shape[0],
                compress="deflate",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic)
        finally:
            for src in srcs:
                src.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
