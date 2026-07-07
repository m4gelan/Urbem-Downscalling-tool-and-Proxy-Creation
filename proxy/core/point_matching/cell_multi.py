from __future__ import annotations

from typing import Any

import numpy as np

from proxy.core import log
from proxy.core.point_matching.scoring import haversine_km, lon_lat_to_cell_ids


def _facility_pollutants(facility: dict[str, Any], pollutant_labels: list[str]) -> dict[str, float]:
    raw = facility.get("pollutants") or {}
    return {lab: float(raw.get(lab, 0.0) or 0.0) for lab in pollutant_labels}


def _total_facility_emissions(
    facilities: list[tuple[str, dict[str, Any]]],
    pollutant_labels: list[str],
) -> dict[str, float]:
    totals = {lab: 0.0 for lab in pollutant_labels}
    for _, fac in facilities:
        fp = _facility_pollutants(fac, pollutant_labels)
        for lab in pollutant_labels:
            totals[lab] += fp[lab]
    return totals


def _avg_facility_shares(
    facilities: list[tuple[str, dict[str, Any]]],
    totals: dict[str, float],
    pollutant_labels: list[str],
) -> tuple[list[str], dict[str, float]]:
    working = [lab for lab in pollutant_labels if totals[lab] > 0.0]
    shares = {fid: 0.0 for fid, _ in facilities}
    if not working:
        return working, shares
    for fid, fac in facilities:
        fac_pols = _facility_pollutants(fac, pollutant_labels)
        shares[fid] = sum(fac_pols[lab] / totals[lab] for lab in working) / len(working)
    return working, shares


def _closest_facility_id(
    facilities: list[tuple[str, dict[str, Any]]],
    cam_lon: float,
    cam_lat: float,
) -> str:
    best_fid = facilities[0][0]
    best_d = float("inf")
    for fid, fac in facilities:
        d = float(
            haversine_km(
                np.array([cam_lon]),
                np.array([cam_lat]),
                np.array([float(fac["lon"])]),
                np.array([float(fac["lat"])]),
            )[0]
        )
        if d < best_d:
            best_d = d
            best_fid = fid
    return best_fid


def allocate_cams_to_facilities(
    cams_row: dict[str, Any],
    facilities: list[tuple[str, dict[str, Any]]],
    pollutant_labels: list[str],
    *,
    proportional: bool,
    cam_lon: float,
    cam_lat: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    cams_pols = {
        lab: float((cams_row.get("pollutants") or {}).get(lab, 0.0) or 0.0)
        for lab in pollutant_labels
    }
    flags: list[str] = []
    if not facilities:
        return [], flags

    if proportional:
        totals = _total_facility_emissions(facilities, pollutant_labels)
        working, avg_shares = _avg_facility_shares(facilities, totals, pollutant_labels)
        closest_fid = _closest_facility_id(facilities, cam_lon, cam_lat) if not working else None
        zero_facility_labels: set[str] = set()
        links: list[dict[str, Any]] = []
        for fid, fac in facilities:
            fac_pols = _facility_pollutants(fac, pollutant_labels)
            attributed: dict[str, float] = {}
            for lab in pollutant_labels:
                cams_val = cams_pols[lab]
                if cams_val <= 0.0:
                    attributed[lab] = 0.0
                    continue
                denom = totals[lab]
                if denom <= 0.0:
                    zero_facility_labels.add(lab)
                    if working:
                        attributed[lab] = cams_val * avg_shares[fid]
                    elif fid == closest_fid:
                        attributed[lab] = cams_val
                    else:
                        attributed[lab] = 0.0
                    continue
                attributed[lab] = cams_val * fac_pols[lab] / denom
            dist = float(
                haversine_km(
                    np.array([cam_lon]),
                    np.array([cam_lat]),
                    np.array([float(fac["lon"])]),
                    np.array([float(fac["lat"])]),
                )[0]
            )
            links.append({
                "facility_id": fid,
                "facility_info": dict(fac),
                "attributed_pollutants": attributed,
                "scoring_value": dist,
            })
        flags.extend(f"zero_facility_total_{lab}" for lab in sorted(zero_facility_labels))
        return links, flags

    dists: list[tuple[float, str, dict[str, Any]]] = []
    for fid, fac in facilities:
        d = float(
            haversine_km(
                np.array([cam_lon]),
                np.array([cam_lat]),
                np.array([float(fac["lon"])]),
                np.array([float(fac["lat"])]),
            )[0]
        )
        dists.append((d, fid, fac))
    dists.sort(key=lambda x: x[0])
    d, fid, fac = dists[0]
    if len(facilities) > 1:
        flags.append("fallback_closest_in_cell")
    attributed = {lab: cams_pols[lab] for lab in pollutant_labels if cams_pols[lab] > 0.0}
    return [{
        "facility_id": fid,
        "facility_info": dict(fac),
        "attributed_pollutants": attributed,
        "scoring_value": d,
    }], flags


def _facilities_by_cell(
    facilities_by_id: dict[str, dict[str, Any]],
    grid_meta: dict[str, Any],
) -> dict[int, list[tuple[str, dict[str, Any]]]]:
    if not facilities_by_id:
        return {}
    ids = list(facilities_by_id.keys())
    lons = np.array([float(facilities_by_id[i]["lon"]) for i in ids], dtype=np.float64)
    lats = np.array([float(facilities_by_id[i]["lat"]) for i in ids], dtype=np.float64)
    cells = lon_lat_to_cell_ids(
        lons,
        lats,
        longitude_bounds=grid_meta["longitude_bounds"],
        latitude_bounds=grid_meta["latitude_bounds"],
        n_longitude=int(grid_meta["n_longitude"]),
        n_latitude=int(grid_meta["n_latitude"]),
    )
    out: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for fid, cell in zip(ids, cells.tolist()):
        if cell < 0:
            continue
        out.setdefault(int(cell), []).append((fid, facilities_by_id[fid]))
    return out


def match_cams_cell_multi(
    cams_points: dict[int, dict[str, Any]],
    facilities_by_id: dict[str, dict[str, Any]],
    *,
    cams_grid_meta: dict[str, Any],
    pollutant_labels: list[str],
    match_source: str,
    proportional: bool,
    same_gnfr: bool = True,
    log_label: str,
) -> dict[int, dict[str, Any]]:
    by_cell = _facilities_by_cell(facilities_by_id, cams_grid_meta)
    out: dict[int, dict[str, Any]] = {}
    n_matched = 0

    for pid, cams_row in cams_points.items():
        cell_id = int(cams_row["cell_id"])
        cell_facs = list(by_cell.get(cell_id, []))
        if same_gnfr:
            gnfr = str(cams_row.get("gnfr") or "").strip()
            if gnfr:
                cell_facs = [
                    (fid, fac) for fid, fac in cell_facs
                    if str(fac.get("gnfr") or "").strip() == gnfr
                ]

        links, flags = allocate_cams_to_facilities(
            cams_row,
            cell_facs,
            pollutant_labels,
            proportional=proportional,
            cam_lon=float(cams_row["longitude"]),
            cam_lat=float(cams_row["latitude"]),
        )
        matched = "yes" if links else "no"
        if matched == "yes":
            n_matched += 1
        out[int(pid)] = {
            "cams": dict(cams_row),
            "matched": matched,
            "match_source": match_source if matched == "yes" else None,
            "facility_links": links,
            "flags": flags,
        }

    log.info(
        f"CAMS-{log_label} cell matching: {n_matched}/{len(cams_points)} CAMS matched "
        f"({'proportional' if proportional else 'closest fallback'})."
    )
    return out


def merge_facility_sources(
    named_sources: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source_name, facs in named_sources.items():
        for fid, row in facs.items():
            key = f"{source_name}:{fid}"
            merged = dict(row)
            merged["facility_id"] = key
            merged["fallback_source"] = source_name
            out[key] = merged
    return out


def apply_fallback_to_unmatched(
    matches: dict[int, dict[str, Any]],
    cams_points: dict[int, dict[str, Any]],
    fallback_facilities: dict[str, dict[str, Any]],
    *,
    cams_grid_meta: dict[str, Any],
    pollutant_labels: list[str],
    match_source: str,
    log_label: str,
) -> None:
    unmatched = {
        pid: cams_points[pid]
        for pid in cams_points
        if matches.get(pid, {}).get("matched") != "yes"
    }
    if not unmatched or not fallback_facilities:
        return
    fb = match_cams_cell_multi(
        unmatched,
        fallback_facilities,
        cams_grid_meta=cams_grid_meta,
        pollutant_labels=pollutant_labels,
        match_source=match_source,
        proportional=False,
        same_gnfr=False,
        log_label=log_label,
    )
    for pid, row in fb.items():
        if row.get("matched") == "yes":
            links = row.get("facility_links") or []
            if links:
                src = links[0].get("facility_info", {}).get("fallback_source")
                if src:
                    row["match_source"] = str(src)
            matches[pid] = row


def log_match_flags(matches: dict[int, dict[str, Any]]) -> None:
    if not log.debug_enabled():
        return
    for pid, row in matches.items():
        flags = row.get("flags") or []
        if not flags:
            continue
        cams = row["cams"]
        log.debug(
            f"CAMS point {pid} ({cams.get('gnfr')}) flags: {', '.join(sorted(set(flags)))}"
        )
