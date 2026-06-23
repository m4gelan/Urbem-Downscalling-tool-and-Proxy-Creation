from __future__ import annotations

from pathlib import Path
from typing import Any

from proxy.core import log
from proxy.core.point_matching.cell_multi import (
    apply_fallback_to_unmatched,
    log_match_flags,
    match_cams_cell_multi,
    merge_facility_sources,
)
from proxy.dataset_loaders.load_cams_points import load_cams_grid_meta
from proxy.dataset_loaders.load_riurbans_points import load_riurbans_points


def _gnfr_sectors_from_cfg(cfg: dict[str, Any]) -> list[str]:
    rus = cfg.get("ri_urbans_point_sources") or {}
    if rus.get("gnfr_sector"):
        return [str(rus["gnfr_sector"]).strip()]
    sectors = rus.get("gnfr_sectors")
    if not isinstance(sectors, list) or not sectors:
        raise ValueError("ri_urbans_point_sources.gnfr_sector or gnfr_sectors is required")
    return [str(s).strip() for s in sectors if str(s).strip()]


def run_sector_point_matching(
    cams_points: dict[int, dict[str, Any]],
    *,
    cfg: dict[str, Any],
    repo_root: Path,
    country_profile: dict[str, str],
    pollutant_labels: list[str],
    cams_nc: Path,
    fallback_sources: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[int, dict[str, Any]]:
    """
    RI-URBANS cell matching first (proportional mass split), then optional merged fallback
    datasets (closest facility in cell when emissions are unavailable).
    """
    if not cams_points:
        raise ValueError("cams_points is empty")

    grid_meta = load_cams_grid_meta(cams_nc)
    filepaths = cfg.get("filepaths") or {}
    ri_path = (filepaths.get("RI-URBANS") or filepaths.get("RI_URBANS") or {}).get("path")
    if not ri_path:
        raise ValueError("filepaths.RI-URBANS.path is required for point matching")

    gnfr_sectors = _gnfr_sectors_from_cfg(cfg)

    log.info("--------------------------------")
    log.info("MATCHING CAMS -> RI-URBANS (cell, proportional)")
    log.info("--------------------------------")
    ri_points = load_riurbans_points(
        repo_root / str(ri_path).replace("\\", "/"),
        country_iso3=country_profile["ISO3"],
        gnfr_sectors=gnfr_sectors,
    )
    matches = match_cams_cell_multi(
        cams_points,
        ri_points,
        cams_grid_meta=grid_meta,
        pollutant_labels=pollutant_labels,
        match_source="riurbans",
        proportional=True,
        same_gnfr=True,
        log_label="RI-URBANS",
    )

    if fallback_sources:
        merged = merge_facility_sources(fallback_sources)
        if merged:
            log.info("--------------------------------")
            log.info("FALLBACK MATCHING (merged facility datasets, closest in cell)")
            log.info("--------------------------------")
            apply_fallback_to_unmatched(
                matches,
                cams_points,
                merged,
                cams_grid_meta=grid_meta,
                pollutant_labels=pollutant_labels,
                match_source="fallback",
                log_label="fallback",
            )

    log_match_flags(matches)
    n_yes = sum(1 for m in matches.values() if m.get("matched") == "yes")
    log.info(f"Point matching total: {n_yes}/{len(matches)} CAMS matched.")
    return matches
