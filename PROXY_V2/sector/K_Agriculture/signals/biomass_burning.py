from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PROXY_V2.core import log
from PROXY_V2.core.raster_helpers import buffer_binary_mask, points_in_mask, rasterize_points_max
from PROXY_V2.dataset_loaders.load_cams_cells_mask import pixels_inside_cams_cells
from PROXY_V2.dataset_loaders.load_viirs import filter_viirs_active_fires, load_viirs_active_fires
from PROXY_V2.sector.K_Agriculture.helper import (
    AgReferenceGrid,
    corine_cropland_mask_on_ref,
    viirs_season,
    z_score_by_cams_cell,
)


@dataclass(frozen=True)
class BiomassBurningResult:
    frp_raster: np.ndarray
    z_scored: np.ndarray
    ref: AgReferenceGrid


def build_biomass_burning(
    repo_root: Path,
    cfg: dict[str, Any],
    *,
    ref: AgReferenceGrid,
    cams_cells: dict[int, dict[str, Any]],
    corine_filepath: str | Path,
    viirs_dir: str | Path,
) -> BiomassBurningResult:
    log.info("--- K_Agriculture signal: biomass burning (3.F) ---")

    corine_cfg = cfg.get("corine") or {}
    viirs_cfg = cfg.get("VIIRS") or {}
    corine_band = int(corine_cfg["band"])
    l3_codes = [int(x) for x in corine_cfg["biomass_burning_l3_codes"]]
    buffer_m = float(viirs_cfg["buffer_m"])

    crop = corine_cropland_mask_on_ref(
        repo_root,
        corine_filepath,
        l3_codes,
        corine_band,
        cams_cells,
        ref_height=ref.height,
        ref_width=ref.width,
        ref_transform=ref.transform,
        ref_crs=ref.crs,
    )
    crop_buf = buffer_binary_mask(crop, ref.transform, buffer_m)
    log.info(
        f"cropland mask: {int((crop > 0).sum())} px, buffered {buffer_m}m: {int((crop_buf > 0).sum())} px"
    )

    fires = load_viirs_active_fires(repo_root / str(viirs_dir).replace("\\", "/"), cams_cells)
    fires = filter_viirs_active_fires(fires, viirs_cfg)

    lon = fires["LONGITUDE"].to_numpy(dtype=np.float64)
    lat = fires["LATITUDE"].to_numpy(dtype=np.float64)
    in_mask = points_in_mask(lon, lat, crop_buf, ref.transform, ref.crs)
    n_in = int(in_mask.sum())
    n_out = int(len(fires) - n_in)
    log.info(f"VIIRS vs buffered cropland: inside={n_in} outside={n_out}")

    if n_in > 0:
        months = pd.to_datetime(fires.loc[in_mask, "ACQ_DATE"]).dt.month
        for season in ("DJF", "MAM", "JJA", "SON"):
            n = int((months.map(viirs_season) == season).sum())
            log.info(f"VIIRS in cropland mask — {season}: {n} points")

    inside = fires.loc[in_mask].copy()
    frp = inside["FRP"].to_numpy(dtype=np.float32)
    frp_r = rasterize_points_max(
        inside["LONGITUDE"].to_numpy(),
        inside["LATITUDE"].to_numpy(),
        frp,
        height=ref.height,
        width=ref.width,
        transform=ref.transform,
        raster_crs=ref.crs,
    )

    cams_px = pixels_inside_cams_cells(
        ref.height, ref.width, ref.transform, ref.crs, cams_cells
    )
    z_inside = cams_px & (ref.nuts_r > 0)
    frp_r = np.where(z_inside, frp_r, 0.0).astype(np.float32)
    z = z_score_by_cams_cell(frp_r, ref.cell_id, inside=z_inside)

    log.info(
        f"biomass burning: FRP max={float(frp_r.max()):.4g} z max={float(z.max()):.4g} "
        f"fire px={int((frp_r > 0).sum())}"
    )
    return BiomassBurningResult(frp_raster=frp_r, z_scored=z, ref=ref)
