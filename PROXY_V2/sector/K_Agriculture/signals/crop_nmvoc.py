from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from PROXY_V2.core import log
from PROXY_V2.writers.debug_dump import Lc1RateSignalDebug
from PROXY_V2.sector.K_Agriculture.helper import (
    AgReferenceGrid,
    build_lc1_rate_surface,
    load_nmvoc_lc1_ef,
)


@dataclass(frozen=True)
class CropNmvocResult:
    rate_kg_nmvoc_ha_yr: np.ndarray
    kg_nmvoc_per_pixel_yr: np.ndarray
    normalized: np.ndarray
    ref: AgReferenceGrid
    rate_debug: Lc1RateSignalDebug | None = None


def build_crop_nmvoc(
    repo_root: Path,
    cfg: dict[str, Any],
    country_profile: dict[str, str],
    *,
    sector_config_path: Path,
    ref: AgReferenceGrid,
    cams_cells: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
    corine_filepath: str | Path,
    lucas_filepath: str | Path,
) -> CropNmvocResult:
    log.info("--- K_Agriculture signal: crop NMVOC (3.Dc) ---")

    ef_path = sector_config_path.parent / "emission_factors.yaml"
    lc1_rates = load_nmvoc_lc1_ef(ef_path)
    dbg_on = log.debug_enabled()
    r, s, norm, rate_dbg = build_lc1_rate_surface(
        repo_root,
        cfg,
        country_profile,
        ref=ref,
        cams_cells=cams_cells,
        cams_grid=cams_grid,
        corine_filepath=corine_filepath,
        lucas_filepath=lucas_filepath,
        lucas_block_key="crop_nmvoc",
        lc1_rates=lc1_rates,
        broad_arable=False,
        log_label="crop NMVOC",
        collect_debug=dbg_on,
        debug_title="Crop NMVOC (3.D)",
        debug_units="kg NMVOC ha-1 yr-1",
    )
    return CropNmvocResult(
        rate_kg_nmvoc_ha_yr=r,
        kg_nmvoc_per_pixel_yr=s,
        normalized=norm,
        ref=ref,
        rate_debug=rate_dbg,
    )
