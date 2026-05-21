from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.ops import unary_union

from PROXY_V2.core import log
from PROXY_V2.core.point_matching.scoring import haversine_km



# Sentinel cost assigned to infeasible pairs (distance > threshold).
# Must be larger than any real distance and larger than the skip-slot cost,
# so the solver never prefers a far pair over leaving a CAMS unmatched.
_INFEASIBLE_PAIR_COST_KM = 1e9


def match_cams_to_facilities_one_to_one(
    cams_points: dict[int, dict[str, Any]],
    facilities_by_id: dict[str, dict[str, Any]],
    *,
    max_match_distance_km: float,
    facility_id_field_in_output_rows: str,
    facility_info_field_in_output_rows: str,
    log_label_for_facility_dataset: str,
) -> dict[int, dict[str, Any]]:
    """
    Match each CAMS point to at most one facility using minimum-cost bipartite assignment
    (Hungarian algorithm), enforcing a strict one-to-one constraint.

    Overview
    --------
    - Each CAMS point is matched to at most one facility, and each facility to at most one CAMS.
    - Only pairs within `max_match_distance_km` (haversine) are considered feasible.
    - The assignment minimises total matched distance across all pairs globally.
    - CAMS points with no feasible facility (or outcompeted for every in-range facility)
      are marked ``matched=no`` but still record their nearest facility for diagnostics.

    Cost matrix design
    ------------------
    The cost matrix has shape (n_cams, n_facilities + n_cams):

        Columns 0 … n_facilities-1   → real facility slots
        Columns n_facilities … end   → one "skip" slot per CAMS row (diagonal)

    Cost ordering ensures correct solver behaviour:

        in-range distance  <  skip cost  <  infeasible sentinel

    This means:
    - The solver prefers a real match when a facility is available and in range.
    - It falls back to "skip" only when every in-range facility is taken by a better global pairing,
      or when no facility is within range at all.

    Parameters
    ----------
    cams_points:
        Dict of CAMS point id → row dict (must contain "longitude" and "latitude").
    facilities_by_id:
        Dict of facility id → row dict (must contain "lon" and "lat").
    max_match_distance_km:
        Maximum haversine distance for a pair to be considered feasible.
    facility_id_field_in_output_rows:
        Key used to store the matched (or nearest) facility id in each output row.
    facility_info_field_in_output_rows:
        Key used to store the matched (or nearest) facility data dict in each output row.
    log_label_for_facility_dataset:
        Human-readable label for the facility dataset, used in log messages.

    Returns
    -------
    Dict of CAMS id → output row dict, each containing:
        - "cams":                            original CAMS row
        - "matched":                         "yes" or "no"
        - facility_id_field_in_output_rows:  matched or nearest facility id
        - facility_info_field_in_output_rows: matched or nearest facility data
        - "scoring_value":                   haversine distance to matched/nearest facility (km)

    Raises
    ------
    SystemExit(1) if inputs are empty or if the solver violates the one-to-one constraint.
    """

    # ------------------------------------------------------------------ #
    # 1. Guard: both inputs must be non-empty to proceed                  #
    # ------------------------------------------------------------------ #
    if not cams_points or not facilities_by_id:
        log.error(f"No CAMS points or no {log_label_for_facility_dataset} rows to match.")
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # 2. Index inputs into ordered lists so matrix rows/cols are stable   #
    # ------------------------------------------------------------------ #
    ordered_cams_ids = list(cams_points.keys())
    ordered_facility_ids = list(facilities_by_id.keys())
    n_cams = len(ordered_cams_ids)
    n_facilities = len(ordered_facility_ids)

    # ------------------------------------------------------------------ #
    # 3. Build the (n_cams × n_facilities) haversine distance matrix      #
    # ------------------------------------------------------------------ #

    # Extract facility coordinates into arrays for vectorised haversine calls.
    facility_lons = np.array([float(facilities_by_id[f]["lon"]) for f in ordered_facility_ids], dtype=np.float64)
    facility_lats = np.array([float(facilities_by_id[f]["lat"]) for f in ordered_facility_ids], dtype=np.float64)

    distance_matrix_km = np.empty((n_cams, n_facilities), dtype=np.float64)

    for row_idx, cams_id in enumerate(ordered_cams_ids):
        cams_row = cams_points[cams_id]
        cam_lon = float(cams_row["longitude"])
        cam_lat = float(cams_row["latitude"])

        # Compute distances from this CAMS point to all facilities at once.
        distance_matrix_km[row_idx, :] = haversine_km(
            np.full(n_facilities, cam_lon),
            np.full(n_facilities, cam_lat),
            facility_lons,
            facility_lats,
        )

    # ------------------------------------------------------------------ #
    # 4. Build the augmented Hungarian cost matrix                        #
    #                                                                     #
    # Shape: (n_cams, n_facilities + n_cams)                              #
    #   - Left block  [: , :n_facilities]  → real facility columns        #
    #   - Right block [: , n_facilities:]  → skip-slot diagonal           #
    # ------------------------------------------------------------------ #

    # Start everything at the infeasible sentinel; we'll overwrite valid entries below.
    n_skip_cols = n_cams  # one exclusive skip slot per CAMS row
    cost_matrix = np.full((n_cams, n_facilities + n_skip_cols), _INFEASIBLE_PAIR_COST_KM, dtype=np.float64)

    # Fill real facility columns: use actual distance if within threshold, sentinel otherwise.
    within_threshold = distance_matrix_km <= max_match_distance_km
    cost_matrix[:, :n_facilities] = np.where(
        within_threshold,
        distance_matrix_km,
        _INFEASIBLE_PAIR_COST_KM,
    )

    # Fill skip-slot diagonal: cost is just above the threshold so any valid match beats it,
    # but it's always cheaper than the infeasible sentinel.
    skip_cost = max_match_distance_km + 1.0
    skip_col_indices = n_facilities + np.arange(n_cams)
    cost_matrix[np.arange(n_cams), skip_col_indices] = skip_cost

    log.debug(f"Hungarian cost matrix:\n{cost_matrix}")

    # ------------------------------------------------------------------ #
    # 5. Solve the assignment problem                                     #
    # ------------------------------------------------------------------ #
    assigned_rows, assigned_cols = linear_sum_assignment(cost_matrix)

    # ------------------------------------------------------------------ #
    # 6. Interpret the assignment results                                 #
    # ------------------------------------------------------------------ #
    out_rows: dict[int, dict[str, Any]] = {}
    n_matched = 0
    facility_match_counts: dict[str, int] = {}

    for row_idx, col_idx in zip(assigned_rows, assigned_cols):
        cams_id = ordered_cams_ids[row_idx]
        cams_row = cams_points[cams_id]

        # Always compute the nearest facility for diagnostics / fallback reporting.
        nearest_col = int(np.argmin(distance_matrix_km[row_idx, :]))
        nearest_facility_id = ordered_facility_ids[nearest_col]
        nearest_distance_km = float(distance_matrix_km[row_idx, nearest_col])

        matched_to_real_facility = col_idx < n_facilities

        if matched_to_real_facility:
            # --- Successful match: solver chose a real facility column ---
            facility_id = ordered_facility_ids[col_idx]
            match_distance_km = float(distance_matrix_km[row_idx, col_idx])

            n_matched += 1
            facility_match_counts[facility_id] = facility_match_counts.get(facility_id, 0) + 1

            log.debug(
                f"CAMS {cams_id} → {log_label_for_facility_dataset} {facility_id}: "
                f"{match_distance_km:.3f} km  [matched=yes]"
            )
            out_rows[cams_id] = {
                "cams": dict(cams_row),
                "matched": "yes",
                facility_id_field_in_output_rows: facility_id,
                facility_info_field_in_output_rows: dict(facilities_by_id[facility_id]),
                "scoring_value": match_distance_km,
            }

        else:
            # --- No match: solver chose the skip slot for this CAMS row ---
            # Record the nearest facility anyway so downstream tools can display it.
            log.debug(
                f"CAMS {cams_id} → {log_label_for_facility_dataset} (unmatched): "
                f"nearest={nearest_facility_id} at {nearest_distance_km:.3f} km  [matched=no]"
            )
            out_rows[cams_id] = {
                "cams": dict(cams_row),
                "matched": "no",
                facility_id_field_in_output_rows: nearest_facility_id,
                facility_info_field_in_output_rows: dict(facilities_by_id[nearest_facility_id]),
                "scoring_value": nearest_distance_km,
            }

    # ------------------------------------------------------------------ #
    # 7. Sanity check: the one-to-one constraint must never be violated   #
    # ------------------------------------------------------------------ #
    duplicate_assignments = {fid: count for fid, count in facility_match_counts.items() if count > 1}
    if duplicate_assignments:
        log.error(
            f"Internal error: the following {log_label_for_facility_dataset} ids were assigned "
            f"to more than one CAMS point: {duplicate_assignments!r}"
        )
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # 8. Summary log                                                      #
    # ------------------------------------------------------------------ #
    log.info(
        f"CAMS–{log_label_for_facility_dataset} one-to-one matching complete: "
        f"{n_matched}/{n_cams} CAMS matched within {max_match_distance_km} km."
    )

    return out_rows



def match_cams_jrc(
    cams_points: dict[int, dict[str, Any]],
    jrc_points: dict[str, dict[str, Any]],
    *,
    max_match_distance_km: float,
) -> dict[int, dict[str, Any]]:
    """Global one-to-one CAMS↔JRC assignment minimizing total km within *max_match_distance_km*."""
    if not cams_points:
        log.error("No CAMS points to match.")
        raise SystemExit(1)

    # Countries with no JRC inventory still run point matching via LCP fallback only.
    if not jrc_points:
        log.info("No JRC units for this country; skipping JRC matching (LCP fallback only).")
        return {
            pid: {"cams": dict(cams_row), "matched": "no", "scoring_value": None}
            for pid, cams_row in cams_points.items()
        }

    return match_cams_to_facilities_one_to_one(
        cams_points,
        jrc_points,
        max_match_distance_km=max_match_distance_km,
        facility_id_field_in_output_rows="jrc_point_id",
        facility_info_field_in_output_rows="jrc_point_info",
        log_label_for_facility_dataset="JRC",
    )


# --------------------------------------------------------------------------- #
# Aviation (GNFR H): OSM aerodrome polygon → match point (PROXY rules)         #
# --------------------------------------------------------------------------- #


def _parse_osm_tags(tags_raw: object) -> dict[str, str]:
    """Normalise the GPKG ``osm_tags`` field: already a dict, or a JSON object string, else empty."""
    if tags_raw is None or (isinstance(tags_raw, float) and math.isnan(tags_raw)):
        return {}
    if isinstance(tags_raw, dict):
        return {str(k): str(v) for k, v in tags_raw.items()}
    s = str(tags_raw).strip()
    if not s:
        return {}
    try:
        d = json.loads(s)
        return {str(k): str(v) for k, v in d.items()} if isinstance(d, dict) else {}
    except json.JSONDecodeError:
        return {}


def _tag_nonempty(tags: dict[str, str], key: str) -> bool:
    """True if tag *key* exists and is non-blank after stripping."""
    v = tags.get(key)
    return v is not None and str(v).strip() != ""


def _scalar_text(val: object) -> str:
    """Single-cell string for names/codes: None / NaN → empty string."""
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except TypeError:
        pass
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def _airport_code_field(row: Any, tags: dict[str, str], key: str) -> str | None:
    """ICAO/IATA from row column or parsed tags; empty → None."""
    s = (_scalar_text(row.get(key)) or _scalar_text(tags.get(key))).upper()
    return s if s else None


def _geom_match_point_m3035(aerodrome_poly: Any, apron_union_clipped: Any | None) -> Any:
    """Pick one Shapely point in **metric CRS** used as the facility location for distance matching.

    Priority (same as PROXY ``aviation_matching``):
    1. If *apron_union_clipped* is non-empty: centroid of that union when it lies inside the union,
       otherwise ``representative_point()`` so the result is always inside the apron fragment.
    2. Else: polygon centroid when it lies inside the aerodrome, else ``representative_point()``
       (covers C-shaped / hollow polygons where the centroid falls outside the ring).
    """
    if apron_union_clipped is not None and not apron_union_clipped.is_empty:
        u = apron_union_clipped
        c = u.centroid
        if u.contains(c):
            return c
        return u.representative_point()
    c0 = aerodrome_poly.centroid
    if aerodrome_poly.contains(c0):
        return c0
    return aerodrome_poly.representative_point()


def _collect_apron_terminal_union(
    aerodrome_poly: Any,
    aux_gpkg: Path,
    layer_names: list[str],
    crs_target: Any,
) -> Any | None:
    """Union of apron/terminal geometries from *layer_names* that hit *aerodrome_poly*, clipped to it."""
    if not layer_names or not aux_gpkg.is_file():
        return None
    parts: list[Any] = []
    for lyr in layer_names:
        aux = gpd.read_file(aux_gpkg, layer=str(lyr))
        if aux.empty:
            continue
        aux = aux.to_crs(crs_target)
        hits = aux[aux.geometry.intersects(aerodrome_poly)]
        if hits.empty:
            continue
        inter = unary_union(hits.geometry).intersection(aerodrome_poly)
        if inter.is_empty:
            continue
        parts.append(inter)
    if not parts:
        return None
    return unary_union(parts)


def build_aviation_osm_facilities_by_id(
    poly_gdf: gpd.GeoDataFrame,
    *,
    apron_gpkg: Path,
    point_matching_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build ``facility_id → {lon, lat, …}`` for :func:`match_cams_to_facilities_one_to_one`.

    *poly_gdf* is already clipped aerodrome polygons in metric CRS. Apron/terminal layers are read
    from *apron_gpkg* (same file as polygons unless the sector config overrides the path).
    """
    pm = point_matching_cfg if isinstance(point_matching_cfg, dict) else {}
    min_km2 = float(pm.get("min_polygon_area_km2", 0.5))
    apron_layers = [str(x) for x in (pm.get("aviation_terminal_apron_layers") or []) if str(x).strip()]

    if poly_gdf.empty:
        return {}

    gdf = poly_gdf
    if gdf.crs is None:
        raise ValueError("OSM polygon GeoDataFrame has no CRS")

    fam_col = "aviation_family" if "aviation_family" in gdf.columns else None
    keep_fam = {"aerodrome", "landuse_aerodrome"}

    out: dict[str, dict[str, Any]] = {}
    used_ids: set[str] = set()

    for row_i, (_, row) in enumerate(gdf.iterrows()):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        tags = _parse_osm_tags(row.get("osm_tags"))

        # Drop military airfields (column when present, else tag ``military=``).
        if fam_col and str(row.get(fam_col, "")).strip().lower() == "military_airfield":
            continue
        if _tag_nonempty(tags, "military"):
            continue
        if fam_col and str(row.get(fam_col, "")).strip().lower() not in keep_fam:
            continue

        # Area in km² on the polygon CRS (expected EPSG:3035, metres → km²).
        area_km2 = float(geom.area) / 1e6
        if area_km2 < min_km2:
            continue

        # Apron/terminal union clipped to polygon → match point in metric CRS → WGS84 lon/lat.
        apron_u = _collect_apron_terminal_union(geom, apron_gpkg, apron_layers, gdf.crs)
        match_pt_m = _geom_match_point_m3035(geom, apron_u)
        match_ll = gpd.GeoDataFrame(geometry=[match_pt_m], crs=gdf.crs).to_crs("EPSG:4326").geometry.iloc[0]

        icao = _airport_code_field(row, tags, "icao")
        iata = _airport_code_field(row, tags, "iata")
        name = _scalar_text(row.get("name")) or _scalar_text(tags.get("name")) or "aerodrome"
        etype = _scalar_text(row.get("osm_element_type")) or "way"
        eid = row.get("osm_element_id")
        if pd.isna(eid) or eid is None:
            eid_i = int(row_i)
        else:
            eid_i = int(eid)

        # Stable id for Hungarian matching: ``osm:<element_type>:<osm_element_id>``.
        fac_id = f"osm:{etype}:{eid_i}"
        if fac_id in used_ids:
            fac_id = f"osm:{etype}:{eid_i}:r{row_i}"
        used_ids.add(fac_id)

        out[fac_id] = {
            "lon": float(match_ll.x),
            "lat": float(match_ll.y),
            "name": name,
            "icao": icao or "",
            "iata": iata or "",
            "osm_element_type": etype,
            "osm_numeric_id": eid_i,
            "area_km2": area_km2,
            "facility_id": fac_id,
        }

    log.info(f"Aviation OSM facility candidates (polygons after filters): {len(out)}")
    return out
