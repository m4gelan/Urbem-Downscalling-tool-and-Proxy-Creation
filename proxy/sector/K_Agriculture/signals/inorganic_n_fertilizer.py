from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from proxy.core import log
from proxy.writers.debug_dump import Lc1RateSignalDebug
from proxy.sector.K_Agriculture.helper import (
    AgReferenceGrid,
    build_lc1_rate_surface,
    load_einarsson_lc1_rates,
)


@dataclass(frozen=True)
class InorganicNFertilizerResult:
    rate_kg_n_ha_yr: np.ndarray
    kg_n_per_pixel_yr: np.ndarray
    normalized: np.ndarray
    ref: AgReferenceGrid
    rate_debug: Lc1RateSignalDebug | None = None


def build_inorganic_n_fertilizer(
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
) -> InorganicNFertilizerResult:
    log.info("--- K_Agriculture signal: inorganic N-fertilizer (3.Da1) ---")

    ef_path = sector_config_path.parent / "emission_factors.yaml"
    lc1_rates = load_einarsson_lc1_rates(ef_path)
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
        lucas_block_key="inorganic_n_fertilizer",
        lc1_rates=lc1_rates,
        broad_arable=True,
        log_label="inorganic N",
        collect_debug=dbg_on,
        debug_title="Inorganic N-fertilizer (3.Da1)",
        debug_units="kg N ha-1 yr-1",
    )
    return InorganicNFertilizerResult(
        rate_kg_n_ha_yr=r,
        kg_n_per_pixel_yr=s,
        normalized=norm,
        ref=ref,
        rate_debug=rate_dbg,
    )
