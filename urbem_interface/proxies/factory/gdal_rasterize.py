from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

logger = logging.getLogger(__name__)


def rasterize_polygons_ones(
    gdf: gpd.GeoDataFrame,
    ref_raster_path: Path,
    output_path: Path,
) -> None:
    """
    Burn 1 into pixels covered by polygons (CRS must match ref raster CRS).
    Uses gdal_rasterize on PATH (no Python osgeo module required).
    """
    ref_raster_path = Path(ref_raster_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = gdf[gdf.geometry.notna()].copy()
    if len(gdf) == 0:
        with rasterio.open(ref_raster_path) as ref:
            prof = ref.profile.copy()
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
            dst.write(np.zeros((dst.height, dst.width), dtype=np.float64), 1)
        logger.info("Rasterized 0 polygons -> %s (empty)", output_path.name)
        return

    with tempfile.TemporaryDirectory(prefix="proxy_factory_") as tmp:
        gpkg = Path(tmp) / "burn.gpkg"
        gdf.to_file(gpkg, driver="GPKG", layer="burn")

        with rasterio.open(ref_raster_path) as r:
            b = r.bounds
            w, h = r.width, r.height
            crs = r.crs.to_string()

        cmd = [
            "gdal_rasterize",
            "-burn",
            "1",
            "-init",
            "0",
            "-a_srs",
            crs,
            "-te",
            str(b.left),
            str(b.bottom),
            str(b.right),
            str(b.top),
            "-ts",
            str(w),
            str(h),
            "-of",
            "GTiff",
            "-co",
            "BIGTIFF=IF_SAFER",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "TILED=YES",
            "-l",
            "burn",
            str(gpkg),
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError(
                "gdal_rasterize not found on PATH. Install GDAL/OSGeo4W or conda-forge gdal."
            ) from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"gdal_rasterize failed: {e.stderr or e.stdout or e}"
            ) from e

        logger.info("Rasterized %d polygons -> %s", len(gdf), output_path.name)
