from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

logger = logging.getLogger(__name__)


def _gdal_rasterize_vector_to_ref(
    vector_path: Path,
    layer: str | None,
    ref_path: Path,
    out_path: Path,
    *,
    burn: float = 1.0,
) -> None:
    with rasterio.open(ref_path) as r:
        b = r.bounds
        w, h = r.width, r.height
        crs = r.crs.to_string()

    cmd = [
        "gdal_rasterize",
        "-burn",
        str(burn),
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
    ]
    if layer:
        cmd.extend(["-l", layer])
    cmd.extend([str(vector_path), str(out_path)])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            "gdal_rasterize not found on PATH; install GDAL binaries or use OSGeo4W."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"gdal_rasterize failed: {e.stderr or e.stdout or e}"
        ) from e
    logger.info("Rasterized vector -> %s", out_path.name)


def build_shipping_proxy_merged_rasters(
    projected_bbox: dict[str, float],
    crs_utm: str,
    crs_wgs: str,
    shipping_shp: Path,
    proxies_folder: Path,
    ref_raster_path: Path,
    out_shipping_path: Path,
    *,
    skip_existing: bool = False,
) -> Path:
    """
    LU_Shipping-style merge: max(lu_ports.tif, rasterized shipping lines).
    Avoids building a per-cell GeoDataFrame (required for continent-scale 100 m).
    """
    proxies_folder = Path(proxies_folder)
    ports_path = proxies_folder / "lu_ports.tif"
    if not ports_path.exists():
        raise FileNotFoundError(f"Missing {ports_path} (build CORINE first)")

    domain_utm = gpd.GeoDataFrame(
        geometry=[
            box(projected_bbox["xmin"], projected_bbox["ymin"], projected_bbox["xmax"], projected_bbox["ymax"])
        ],
        crs=crs_utm,
    )
    domain_wgs = domain_utm.to_crs(crs_wgs)
    wgs_bounds = domain_wgs.total_bounds
    wgs_bbox = (wgs_bounds[0], wgs_bounds[1], wgs_bounds[2], wgs_bounds[3])

    ships_gdf = gpd.read_file(str(shipping_shp), bbox=wgs_bbox)
    lines_tmp = proxies_folder / "_shipping_routes_3035.gpkg"
    lines_ras = proxies_folder / "_shipping_lines.tif"

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

    out_shipping_path = Path(out_shipping_path)
    out_shipping_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and out_shipping_path.is_file():
        logger.info("Shipping skipped (--skip-existing): %s exists", out_shipping_path.name)
        return out_shipping_path

    if len(ships_gdf) == 0:
        logger.warning("No shipping routes in bbox; copying ports raster to shipping output.")
        with rasterio.open(ports_path) as src, rasterio.open(out_shipping_path, "w", **prof) as dst:
            for _, window in dst.block_windows(1):
                blk = src.read(1, window=window).astype(np.float64)
                dst.write(blk, 1, window=window)
        return out_shipping_path

    ships_3035 = ships_gdf.to_crs(crs_utm)
    ships_3035.to_file(lines_tmp, driver="GPKG", layer="routes")
    _gdal_rasterize_vector_to_ref(lines_tmp, "routes", ref_raster_path, lines_ras, burn=1.0)

    with rasterio.open(ports_path) as ports, rasterio.open(lines_ras) as lines, rasterio.open(
        out_shipping_path, "w", **prof
    ) as dst:
        if ports.shape != lines.shape:
            raise ValueError(f"Shape mismatch ports {ports.shape} vs lines {lines.shape}")
        for _, window in dst.block_windows(1):
            p = ports.read(1, window=window).astype(np.float64)
            ln = lines.read(1, window=window).astype(np.float64)
            m = np.maximum(p, ln)
            dst.write(m, 1, window=window)

    logger.info("Shipping proxy written: %s", out_shipping_path.name)
    return out_shipping_path
