from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from urbem_interface.proxies.factory.reference import TemplateGrid

logger = logging.getLogger(__name__)

_PROXY_FILENAME: dict[str, str] = {
    "urban": "lu_urban.tif",
    "ind": "lu_industry.tif",
    "agr": "lu_agriculture.tif",
    "airport": "lu_airport.tif",
    "ports": "lu_ports.tif",
    "offroad": "lu_offroad.tif",
    "snap8_temp": "lu_snap8_temp.tif",
}


def expected_corine_output_paths(proxies_folder: Path, corine_classes: dict[str, Any]) -> list[Path]:
    """Paths that a full CORINE phase writes (for --skip-existing checks)."""
    proxies_folder = Path(proxies_folder)
    proxy_codes = {k: v for k, v in corine_classes["proxy_reclassified_code"].items() if not k.startswith("_")}
    paths: list[Path] = [
        proxies_folder / "corine_utm.tif",
        proxies_folder / "corine_reclass.tif",
    ]
    for proxy_name in proxy_codes:
        fn = _PROXY_FILENAME.get(proxy_name)
        if fn:
            paths.append(proxies_folder / fn)
    return paths


def corine_phase_complete(proxies_folder: Path, corine_classes: dict[str, Any]) -> bool:
    return all(p.is_file() for p in expected_corine_output_paths(proxies_folder, corine_classes))


def _warp_corine_to_grid(
    corine_raster_path: Path,
    out_path: Path,
    grid: TemplateGrid,
) -> None:
    dst_transform = grid.transform
    dst_crs = grid.crs
    dst_w = grid.width
    dst_h = grid.height
    dst_nodata = -9999.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(corine_raster_path) as src:
        src_nodata = src.nodata
        kwargs: dict[str, Any] = {
            "driver": "GTiff",
            "width": dst_w,
            "height": dst_h,
            "count": 1,
            "dtype": "float64",
            "crs": dst_crs,
            "transform": dst_transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
            "nodata": dst_nodata,
        }
        with rasterio.open(out_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                src_nodata=src_nodata,
                dst_nodata=dst_nodata,
            )
    logger.info("CORINE warped to grid: %s", out_path)


def _reclassify_block(raw: np.ndarray, reclassification: dict[str, Any]) -> np.ndarray:
    orig = raw.copy()
    reclass = np.array(raw, copy=True)
    for clc_str, class_num in reclassification.items():
        if clc_str.startswith("_"):
            continue
        clc_code = int(clc_str)
        reclass[orig == clc_code] = class_num
    return reclass


def _write_reclassified_tif(src_path: Path, dst_path: Path, reclassification: dict[str, Any]) -> None:
    with rasterio.open(src_path) as src:
        prof = src.profile.copy()
        prof.update(
            dtype="float64",
            nodata=None,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(dst_path, "w", **prof) as dst:
            for _, window in src.block_windows(1):
                raw = src.read(1, window=window)
                rc = _reclassify_block(raw, reclassification).astype(np.float64)
                dst.write(rc, 1, window=window)
    logger.info("Wrote reclassified CORINE: %s", dst_path.name)


def _write_binary_from_reclass(
    reclass_path: Path,
    class_codes: list[int] | int,
    output_path: Path,
) -> None:
    if isinstance(class_codes, int):
        class_codes = [class_codes]

    with rasterio.open(reclass_path) as src:
        prof = src.profile.copy()
        prof.update(
            dtype="float64",
            nodata=None,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="IF_SAFER",
        )
        with rasterio.open(output_path, "w", **prof) as dst:
            for _, window in src.block_windows(1):
                base = src.read(1, window=window).astype(np.float64)
                binary = np.zeros_like(base, dtype=np.float64)
                binary[np.isin(base, class_codes)] = 1.0
                dst.write(binary, 1, window=window)

    n_one = 0
    with rasterio.open(output_path) as chk:
        for _, window in chk.block_windows(1):
            n_one += int(np.sum(chk.read(1, window=window) >= 0.5))
    logger.info("Binary proxy %s (cells ~1): %d", output_path.name, n_one)


def build_corine_proxies_on_grid(
    corine_raster_path: Path,
    proxies_folder: Path,
    grid: TemplateGrid,
    corine_classes: dict[str, Any],
    *,
    skip_existing: bool = False,
) -> dict[str, Path]:
    proxies_folder = Path(proxies_folder)
    proxies_folder.mkdir(parents=True, exist_ok=True)

    reclassification = {k: v for k, v in corine_classes["reclassification"].items() if not k.startswith("_")}
    proxy_codes = {k: v for k, v in corine_classes["proxy_reclassified_code"].items() if not k.startswith("_")}

    corine_warped = proxies_folder / "corine_utm.tif"
    corine_reclass = proxies_folder / "corine_reclass.tif"

    if skip_existing and corine_phase_complete(proxies_folder, corine_classes):
        logger.info("CORINE phase skipped (--skip-existing): all expected rasters already in %s", proxies_folder)
        written: dict[str, Path] = {}
        for proxy_name in proxy_codes:
            fn = _PROXY_FILENAME.get(proxy_name)
            if fn:
                written[proxy_name] = proxies_folder / fn
        written["corine_utm"] = corine_warped
        written["corine_reclass"] = corine_reclass
        return written

    _warp_corine_to_grid(corine_raster_path, corine_warped, grid)
    _write_reclassified_tif(corine_warped, corine_reclass, reclassification)

    written: dict[str, Path] = {"corine_utm": corine_warped, "corine_reclass": corine_reclass}

    for proxy_name, class_codes in proxy_codes.items():
        filename = _PROXY_FILENAME.get(proxy_name)
        if filename is None:
            logger.warning("No filename mapping for proxy '%s' — skipping.", proxy_name)
            continue
        out_path = proxies_folder / filename
        _write_binary_from_reclass(corine_reclass, class_codes, out_path)
        written[proxy_name] = out_path

    gc.collect()
    logger.info("CORINE proxies on grid: %d rasters", len(written))
    return written
