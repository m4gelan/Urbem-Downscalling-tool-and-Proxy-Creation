from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from PROXY_V2.core import log
from PROXY_V2.core.raster_helpers import cams_cell_id_for_raster, restrict_cell_ids_to_country
from PROXY_V2.dataset_loaders.load_cams_cells_mask import (
    pixels_inside_cams_cells,
    read_raster_window_for_cams,
)


def load_emodnet(
    emodnet_filepath: Path,
    cams_cells: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
    *,
    band: int,
) -> tuple[np.ndarray, Any, Any, np.ndarray]:
    """
    Vessel-density raster clipped to the CAMS cell union window.

    Pixels outside the CAMS footprint are set to ``0``. ``cell_id`` matches the
    CORINE / CAMS convention on this raster's grid (``-1`` outside country cells).
    """
    raw, transform, raster_crs, nodata = read_raster_window_for_cams(
        emodnet_filepath, band, cams_cells
    )

    data = np.asarray(raw, dtype=np.float32)
    if nodata is not None:
        data = np.where(np.asarray(raw) == nodata, 0.0, data)
    data = np.where(np.isfinite(data), data, 0.0)

    h, w = data.shape
    inside = pixels_inside_cams_cells(h, w, transform, raster_crs, cams_cells)
    out = (data * inside).astype(np.float32, copy=False)

    px_w, px_h = abs(float(transform.a)), abs(float(transform.e))
    log.info(
        f"EMODNET {emodnet_filepath.name}: shape={w}×{h} band={band} "
        f"pixel_size=({px_w:.8g}, {px_h:.8g}) CRS units, crs={raster_crs}"
    )
    vals = out[inside]
    if vals.size:
        log.info(
            "EMODNET values (pixel centres inside CAMS cells): "
            f"min={float(np.min(vals)):.6g} max={float(np.max(vals)):.6g} "
            f"mean={float(np.mean(vals)):.6g} median={float(np.median(vals)):.6g}"
        )
    else:
        log.info("EMODNET: no pixels inside CAMS cell footprint on this window")

    nlon = int(cams_grid["n_longitude"])
    nlat = int(cams_grid["n_latitude"])
    cell_id = cams_cell_id_for_raster(
        transform,
        raster_crs,
        h,
        w,
        cams_grid["lon_bounds"],
        cams_grid["lat_bounds"],
        nlon,
        nlat,
    )
    cell_id = restrict_cell_ids_to_country(cell_id, cams_cells)
    log.info(f"EMODNET: {int((out > 0).sum())} pixels > 0 inside window")

    return out, transform, raster_crs, cell_id
