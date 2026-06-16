from __future__ import annotations

from pathlib import Path
from typing import Any

from proxy.core import log
from proxy.core.point_matching.matching import match_cams_to_facilities_one_to_one
from proxy.dataset_loaders.load_corine import get_corine_airport_facilities


def merge_corine_aviation_fallback(
    matches: dict[int, dict[str, Any]],
    cor_fb: dict[int, dict[str, Any]],
) -> None:
    """In-place: fill ``matched`` from CORINE Hungarian rows where OSM left a CAMS unmatched."""
    for pid, cm in cor_fb.items():
        if matches[pid].get("matched") == "yes":
            continue
        if cm.get("matched") != "yes":
            continue
        prev = matches[pid]
        matches[pid] = {
            "cams": prev["cams"],
            "matched": "yes",
            "match_source": "corine",
            "corine_facility_id": cm["corine_facility_id"],
            "corine_facility_info": cm["corine_facility_info"],
            "scoring_value": cm["scoring_value"],
            "osm_facility_id": prev.get("osm_facility_id"),
            "osm_facility_info": prev.get("osm_facility_info"),
        }


def match_cams_lcp_one_to_one(
    cams_without_jrc_match: dict[int, dict[str, Any]],
    lcp_facilities_by_id: dict[str, dict[str, Any]],
    *,
    match_mode: str,
    max_match_distance_km: float | None = None,
    cams_grid_meta: dict[str, Any] | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Second stage: CAMS left after JRC, matched uniquely to **LCP** installations
    (same global one-to-one rule as JRC). Output rows reuse ``eprtr_*`` keys so the
    rest of the pipeline (map / GeoTIFF) stays unchanged.
    """
    log.info(f"LCP fallback: {len(cams_without_jrc_match)} CAMS without a JRC match")
    if not lcp_facilities_by_id:
        log.info("No LCP facilities for this country; CAMS remain unmatched after fallback.")
        return {}

    fallback_rows = match_cams_to_facilities_one_to_one(
        cams_without_jrc_match,
        lcp_facilities_by_id,
        match_mode=match_mode,
        max_match_distance_km=max_match_distance_km,
        cams_grid_meta=cams_grid_meta,
        facility_id_field_in_output_rows="eprtr_point_id",
        facility_info_field_in_output_rows="eprtr_point_info",
        log_label_for_facility_dataset="LCP",
    )
    for row in fallback_rows.values():
        row["match_layer"] = "lcp_fallback"
    return fallback_rows


def merge_uwwtd_waste_fallback(
    matches: dict[int, dict[str, Any]],
    uww_fb: dict[int, dict[str, Any]],
) -> None:
    """In-place: fill ``matched`` from UWWTD Hungarian rows where E-PRTR left a CAMS unmatched."""
    for pid, um in uww_fb.items():
        if pid not in matches:
            continue
        if matches[pid].get("matched") == "yes":
            continue
        if um.get("matched") != "yes":
            continue
        prev = matches[pid]
        matches[pid] = {
            "cams": prev["cams"],
            "matched": "yes",
            "match_source": "uwwtd",
            "uwwtd_facility_id": um["uwwtd_facility_id"],
            "uwwtd_facility_info": um["uwwtd_facility_info"],
            "scoring_value": um["scoring_value"],
            "eprtr_point_id": prev.get("eprtr_point_id"),
            "eprtr_point_info": prev.get("eprtr_point_info"),
        }
