"""
Utilities for the line-sources pipeline.

Replicates R's areasources_to_osm_linesources.R: OSM road fetch and
VEIN-inspired area-to-line emission allocation.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import box

from urbem_interface.logging_config import get_logger
from urbem_interface.utils.domain import domain_bounds_wgs84

logger = get_logger(__name__)

POLLUTANTS_OUT = ["NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"]

domain_bounds_wgs84_from_cfg = domain_bounds_wgs84

import inspect
_OSM_BBOX_NEW_API: bool = "bbox" in inspect.signature(ox.graph_from_bbox).parameters


def fetch_osm_roads(
    bbox_wgs84: tuple[float, float, float, float],
    road_types: list[str],
    target_crs: str | Any,
) -> gpd.GeoDataFrame:
    """
    Fetch OSM road network in bbox, filter by highway types, reproject to target CRS.

    bbox_wgs84 : (xmin, ymin, xmax, ymax) in WGS84 lon/lat.
    road_types : OSM highway tag values e.g. ["motorway", "trunk", "primary"].
    target_crs : CRS to reproject edges into (domain CRS).

    Returns GeoDataFrame with columns ["highway", "geometry"] in target_crs.
    Empty GeoDataFrame (same columns, target_crs) if nothing found.
    """
    _EMPTY = gpd.GeoDataFrame(columns=["highway", "geometry"]).set_crs(target_crs)

    if not road_types:
        logger.warning("fetch_osm_roads: road_types is empty - returning empty GeoDataFrame")
        return _EMPTY

    custom_filter = f'["highway"~"^({"|".join(road_types)})$"]'
    xmin, ymin, xmax, ymax = bbox_wgs84

    logger.info(
        f"Fetching OSM roads  bbox=({xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f})  "
        f"types={road_types}"
    )

    try:
        if _OSM_BBOX_NEW_API:
            G = ox.graph_from_bbox(bbox_wgs84, custom_filter=custom_filter, simplify = False)
        else:
            # osmnx < 1.7 expects (north, south, east, west)
            G = ox.graph_from_bbox(ymax, ymin, xmax, xmin, custom_filter=custom_filter, simplify = False)
    except Exception as e:
        logger.error(f"OSM download failed: {e}")
        raise

    edges = ox.graph_to_gdfs(G, nodes=False, edges=True, fill_edge_geometry=True)

    if edges.empty:
        logger.warning("OSM returned no edges for this domain")
        return _EMPTY

    edges = edges[["highway", "geometry"]].copy()
    edges = edges[edges["geometry"].notna()].copy()

    if edges.empty:
        logger.warning("All OSM edges dropped after geometry filter")
        return _EMPTY

    # Flatten list-valued highway tags (osmnx sometimes returns lists)
    edges["highway"] = edges["highway"].apply(
        lambda h: h[0] if isinstance(h, (list, tuple)) and h else (str(h) if h else "")
    )

    edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=4326).to_crs(target_crs)

    logger.info(f"  -> {len(edges):,} road segments after filtering and reprojection")
    return edges


def _cell_polygon(
    row: int,
    col: int,
    transform: Any,
) -> Any:
    """
    Build Shapely box for raster cell (row, col).

    Works correctly for north-up rasters where transform.e is negative.
    nrow/ncol args removed - not needed for single-cell construction.
    """
    x_left  = transform.c + col * transform.a
    x_right = transform.c + (col + 1) * transform.a
    y_top   = transform.f + row * transform.e
    y_bot   = transform.f + (row + 1) * transform.e
    # min/max guards against both north-up (e<0) and south-up (e>0) rasters
    return box(min(x_left, x_right), min(y_top, y_bot),
               max(x_left, x_right), max(y_top, y_bot))


def area_to_osm_lines(
    domain_transform: Any,
    domain_shape: tuple[int, int],
    domain_crs: Any,
    emissions: dict[str, np.ndarray],
    osm_gdf: gpd.GeoDataFrame,
    road_type_weights: dict[str, float],
    *,
    split_by_cell: bool = False,
    road_grouping: str = "config",
    include_zero_emissions_cells: bool = False,
) -> gpd.GeoDataFrame:
    """
    Distribute gridded area emissions onto OSM road segments.

    For each raster cell:
      - Clips road segments to the cell polygon (via spatial index).
      - Weights emission by road category and segment length fraction.
      - If `split_by_cell=False`: accumulate emissions back onto the
        original OSM segment id (like the previous implementation).
      - If `split_by_cell=True`: emit one output row per clipped road piece
        (like the R `st_intersection()` approach).

    Returns GeoDataFrame with columns = ["geometry", "roadtype"] + POLLUTANTS_OUT.
    """
    nrow, ncol = domain_shape
    road_grouping = str(road_grouping).lower().strip()

    # "r4" mimics the R allocation groups:
    #   motorway = motorway + motorway_link (grep("motorway", ...))
    #   trunk    = trunk + trunk_link
    #   primary  = primary + primary_link
    #   secondary= secondary + secondary_link
    if road_grouping == "r4":
        group_defs = [
            ("motorway", "motorway"),
            ("trunk", "trunk"),
            ("primary", "primary"),
            ("secondary", "secondary"),
        ]

        def _w_for(pattern: str) -> float:
            matches = [
                float(w)
                for k, w in road_type_weights.items()
                if pattern in str(k).lower()
            ]
            return max(matches) if matches else 1.0

        group_weights = {group: _w_for(pattern) for group, pattern in group_defs}
        # Keep just the patterns; group label is used as the conceptual bucket.
        group_patterns = {group: pattern for group, pattern in group_defs}
    else:
        # Config-driven categories (previous behavior).
        categories = list(road_type_weights.keys())

    if osm_gdf.empty:
        logger.warning("area_to_osm_lines: osm_gdf is empty - returning empty result")
        return gpd.GeoDataFrame(columns=["geometry", "roadtype"] + POLLUTANTS_OUT, crs=domain_crs)

    # Compute full segment lengths once (in domain CRS metres)
    osm_gdf = osm_gdf.copy()
    osm_gdf["_length_m"] = osm_gdf.geometry.length
    osm_gdf["_highway_lower"] = osm_gdf["highway"].astype(str).str.lower()

    # Accumulator: seg_idx -> poll -> value (only used when split_by_cell=False)
    seg_emit: dict[int, dict[str, float]] = {} if not split_by_cell else {}
    result_rows: list[dict[str, Any]] | None = [] if split_by_cell else None

    sindex = osm_gdf.sindex
    active_polls = [p for p in POLLUTANTS_OUT if p in emissions]

    logger.info(f"Distributing emissions onto {len(osm_gdf):,} OSM segments  "
                f"grid={nrow}x{ncol}  polls={active_polls}")

    total_cells = nrow * ncol
    log_every   = max(1, total_cells // 10)

    for idx_cell, (row, col) in enumerate(
        (r, c) for r in range(nrow) for c in range(ncol)
    ):
        if idx_cell % log_every == 0:
            logger.debug(f"  cell {idx_cell}/{total_cells} ({100*idx_cell//total_cells}%)")

        # Cell emissions:
        # - default: only consider pollutants with value > 0 (fast path)
        # - align-to-R mode: consider also zeros (so we emit zero-emission road clips)
        if include_zero_emissions_cells:
            cell_vals = {}
            for p in active_polls:
                v = emissions[p][row, col]
                if np.isnan(v):
                    continue
                cell_vals[p] = float(v)
        else:
            cell_vals = {
                p: float(v)
                for p in active_polls
                if (v := emissions[p][row, col]) > 0 and not np.isnan(v)
            }

        if not cell_vals:
            continue

        cell_poly = _cell_polygon(row, col, domain_transform)

        # Candidate segments via spatial index
        candidates_idx = list(sindex.intersection(cell_poly.bounds))
        if not candidates_idx:
            continue

        candidates = osm_gdf.iloc[candidates_idx].copy()

        # Clip to cell
        clipped = candidates.copy()
        clipped.geometry = candidates.geometry.intersection(cell_poly)
        clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
        if clipped.empty:
            continue

        clipped["_clip_len"] = clipped.geometry.length

        # Distribute with either R-like grouping ("r4") or config categories
        if road_grouping == "r4":
            # Determine which groups are present in this clipped set
            present_weights: dict[str, float] = {}
            group_masks: dict[str, Any] = {}
            for group, pattern in group_patterns.items():
                mask = clipped["_highway_lower"].str.contains(pattern, regex=False, na=False)
                group_masks[group] = mask
                if mask.any():
                    present_weights[group] = group_weights[group]

            total_w = sum(present_weights.values())
            if total_w <= 0:
                continue

            # When splitting by cell, we update emissions on the clipped rows directly.
            if split_by_cell and result_rows is not None:
                for poll in POLLUTANTS_OUT:
                    clipped[poll] = 0.0

            for group, w in present_weights.items():
                norm_w = w / total_w
                sub_mask = group_masks[group]
                sub = clipped[sub_mask]
                if sub.empty:
                    continue

                total_len = float(sub["_clip_len"].sum())
                if total_len <= 0:
                    continue

                fracs = sub["_clip_len"] / total_len  # aligned Series
                for poll, cell_val in cell_vals.items():
                    emit_per_seg = fracs * norm_w * cell_val  # aligned Series
                    if split_by_cell and result_rows is not None:
                        # Add to clipped rows (indexed by sub.index)
                        clipped.loc[sub.index, poll] = (
                            clipped.loc[sub.index, poll] + emit_per_seg
                        )
                    else:
                        # Accumulate onto original OSM segment id
                        for orig_idx, emit in zip(sub.index, emit_per_seg):
                            if orig_idx not in seg_emit:
                                seg_emit[orig_idx] = {p: 0.0 for p in POLLUTANTS_OUT}
                            seg_emit[orig_idx][poll] += float(emit)

            if split_by_cell and result_rows is not None:
                # Emit one row per clipped road piece
                clipped_out = clipped[["geometry", "highway"] + list(cell_vals.keys())].copy()
                clipped_out = clipped_out.rename(columns={"highway": "roadtype"})
                # Ensure all pollutant columns exist
                for poll in POLLUTANTS_OUT:
                    if poll not in clipped_out.columns:
                        clipped_out[poll] = 0.0
                # Store as dicts to avoid GeoDataFrame concat quirks
                out_dicts: list[dict[str, Any]] = []
                for idx in clipped_out.index:
                    row = {poll: float(clipped_out.at[idx, poll]) for poll in POLLUTANTS_OUT}
                    row["roadtype"] = str(clipped_out.at[idx, "roadtype"])
                    row["geometry"] = clipped_out.at[idx, "geometry"]
                    out_dicts.append(row)
                result_rows.extend(out_dicts)

        else:
            # Config-driven categories (previous behavior)
            cat_weights: dict[str, float] = {}
            cat_masks: dict[str, Any] = {}
            for cat in categories:
                mask = clipped["_highway_lower"].str.contains(cat, regex=False, na=False)
                cat_masks[cat] = mask
                if mask.any():
                    cat_weights[cat] = road_type_weights.get(cat, 1.0)

            total_w = sum(cat_weights.values())
            if total_w <= 0:
                continue
            norm_weights = {cat: w / total_w for cat, w in cat_weights.items()}

            for cat, norm_w in norm_weights.items():
                mask = cat_masks[cat]
                sub = clipped[mask]
                if sub.empty:
                    continue

                total_len = float(sub["_clip_len"].sum())
                if total_len <= 0:
                    continue

                fracs = sub["_clip_len"] / total_len  # aligned Series
                for poll, cell_val in cell_vals.items():
                    emit_per_seg = fracs * norm_w * cell_val  # aligned Series

                    if split_by_cell and result_rows is not None:
                        # Initialize once for this cell
                        if poll in cell_vals:
                            if poll not in clipped.columns:
                                clipped[poll] = 0.0
                        clipped.loc[sub.index, poll] = (
                            clipped.loc[sub.index, poll] + emit_per_seg
                        )
                    else:
                        for orig_idx, emit in zip(sub.index, emit_per_seg):
                            if orig_idx not in seg_emit:
                                seg_emit[orig_idx] = {p: 0.0 for p in POLLUTANTS_OUT}
                            seg_emit[orig_idx][poll] += float(emit)

            if split_by_cell and result_rows is not None:
                clipped_out = clipped[["geometry", "highway"] + list(POLLUTANTS_OUT)].copy().rename(
                    columns={"highway": "roadtype"}
                )
                # Ensure all pollutant columns exist even if we never updated them in this cell
                for poll in POLLUTANTS_OUT:
                    if poll not in clipped_out.columns:
                        clipped_out[poll] = 0.0
                out_dicts: list[dict[str, Any]] = []
                for idx in clipped_out.index:
                    row = {poll: float(clipped_out.at[idx, poll]) for poll in POLLUTANTS_OUT}
                    row["roadtype"] = str(clipped_out.at[idx, "roadtype"])
                    row["geometry"] = clipped_out.at[idx, "geometry"]
                    out_dicts.append(row)
                result_rows.extend(out_dicts)

    if split_by_cell:
        if not result_rows:
            logger.warning("area_to_osm_lines: no clipped road pieces produced")
            return gpd.GeoDataFrame(columns=["geometry", "roadtype"] + POLLUTANTS_OUT, crs=domain_crs)
        out = gpd.GeoDataFrame(result_rows, geometry="geometry", crs=domain_crs)
    else:
        if not seg_emit:
            logger.warning("area_to_osm_lines: no emissions distributed to any segment")
            return gpd.GeoDataFrame(columns=["geometry", "roadtype"] + POLLUTANTS_OUT, crs=domain_crs)

        result_rows_final: list[dict[str, Any]] = []
        for orig_idx, emit_dict in seg_emit.items():
            seg = osm_gdf.loc[orig_idx]
            row_data = {p: emit_dict.get(p, 0.0) for p in POLLUTANTS_OUT}
            row_data["geometry"] = seg.geometry
            row_data["roadtype"] = seg["highway"]
            result_rows_final.append(row_data)
        out = gpd.GeoDataFrame(result_rows_final, geometry="geometry", crs=domain_crs)

    total_nox = float(out["NOx"].sum()) if "NOx" in out.columns else float("nan")
    logger.info(f"  -> {len(out):,} segments with emissions  NOx_total={total_nox:.2f}")
    return out


def line_start_end_coords(
    gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (x_start, y_start, x_end, y_end) arrays from line geometries.

    Handles LineString, MultiLineString, degenerate single-point lines.
    NaN is returned for missing/empty geometries or degenerate cases.
    """
    from shapely.geometry import MultiLineString

    n = len(gdf)
    x_start = np.full(n, np.nan)
    y_start = np.full(n, np.nan)
    x_end   = np.full(n, np.nan)
    y_end   = np.full(n, np.nan)

    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue  # stays NaN

        if isinstance(geom, MultiLineString):
            lines = geom.geoms
            if not lines:
                continue
            start_pt = lines[0].coords[0]
            end_pt   = lines[-1].coords[-1]

        else:
            coords = list(geom.coords)
            if not coords:
                continue
            start_pt = coords[0]
            end_pt   = coords[-1] if len(coords) >= 2 else coords[0]

        x_start[i], y_start[i] = start_pt[0], start_pt[1]
        x_end[i],   y_end[i]   = end_pt[0],   end_pt[1]

    return x_start, y_start, x_end, y_end
