from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features as rio_features
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping, shape as shp_shape

from PROXY_V2.core import log
from PROXY_V2.core.point_matching.matching import _geom_match_point_m3035
from PROXY_V2.core.raster_helpers import (
    cams_cell_id_for_raster,
    restrict_cell_ids_to_country,
)
from PROXY_V2.dataset_loaders.load_cams_cells_mask import (
    pixels_inside_cams_cells,
    read_raster_window_for_cams,
)

# Raster pixel value (1..44) → (CLC level-3 code, human label)
_CORINE_CLASSES: tuple[tuple[int, int, str], ...] = (
    (1,  111, "Continuous urban fabric"),
    (2,  112, "Discontinuous urban fabric"),
    (3,  121, "Industrial or commercial units"),
    (4,  122, "Road and rail networks and associated land"),
    (5,  123, "Port areas"),
    (6,  124, "Airports"),
    (7,  131, "Mineral extraction sites"),
    (8,  132, "Dump sites"),
    (9,  133, "Construction sites"),
    (10, 141, "Green urban areas"),
    (11, 142, "Sport and leisure facilities"),
    (12, 211, "Non-irrigated arable land"),
    (13, 212, "Permanently irrigated land"),
    (14, 213, "Rice fields"),
    (15, 221, "Vineyards"),
    (16, 222, "Fruit trees and berry plantations"),
    (17, 223, "Olive groves"),
    (18, 231, "Pastures"),
    (19, 241, "Annual crops associated with permanent crops"),
    (20, 242, "Complex cultivation patterns"),
    (21, 243, "Land principally occupied by agriculture with significant natural vegetation"),
    (22, 244, "Agro-forestry areas"),
    (23, 311, "Broad-leaved forest"),
    (24, 312, "Coniferous forest"),
    (25, 313, "Mixed forest"),
    (26, 321, "Natural grasslands"),
    (27, 322, "Moors and heathland"),
    (28, 323, "Sclerophyllous vegetation"),
    (29, 324, "Transitional woodland-shrub"),
    (30, 331, "Beaches dunes sands"),
    (31, 332, "Bare rocks"),
    (32, 333, "Sparsely vegetated areas"),
    (33, 334, "Burnt areas"),
    (34, 335, "Glaciers and perpetual snow"),
    (35, 411, "Inland marshes"),
    (36, 412, "Peat bogs"),
    (37, 421, "Salt marshes"),
    (38, 422, "Salines"),
    (39, 423, "Intertidal flats"),
    (40, 511, "Water courses"),
    (41, 512, "Water bodies"),
    (42, 521, "Coastal lagoons"),
    (43, 522, "Estuaries"),
    (44, 523, "Sea and ocean"),
)

_L3_TO_INDEX: dict[int, int] = {l3: idx for idx, l3, _ in _CORINE_CLASSES}
_L3_TO_LABEL: dict[int, str] = {l3: lab for _, l3, lab in _CORINE_CLASSES}
_INDEX_TO_L3: dict[int, int] = {idx: l3 for idx, l3, _ in _CORINE_CLASSES}


def corine_codes_matching(corine_codes: list[int], corine_band: int) -> np.ndarray:
    """Return sorted unique raster-pixel indices for the given L3 codes."""
    _ = corine_band
    try:
        indices = {_L3_TO_INDEX[int(c)] for c in corine_codes}
    except KeyError as e:
        raise ValueError(f"Unknown CORINE L3 code {e.args[0]!r}") from None
    return np.fromiter(sorted(indices), dtype=np.int64)


def load_corine(
    corine_filepath: Path,
    corine_codes: list[int],
    corine_band: int,
    cams_cells: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
    *,
    need_cell_id: bool = True,
    return_l3: bool = False,
) -> tuple[np.ndarray, Any, Any, np.ndarray]:
    """
    CORINE window over the CAMS footprint on the reference grid.

    Default: float32 mask (1 = pixel matches one of *corine_codes*, else 0).
    With ``return_l3=True``: int16 raster of CLC level-3 codes (0 outside *corine_codes*).

    When ``need_cell_id`` is true, returns ``cell_id`` (int32): flat CAMS index per pixel, -1 outside.
    """
    want_l3 = {int(c) for c in corine_codes}
    unknown = want_l3 - _L3_TO_LABEL.keys()
    if unknown:
        raise ValueError(f"Unknown CORINE L3 code(s): {sorted(unknown)}")

    log.info(
        "CORINE land covers in use:\n"
        + "\n".join(f"  {l3} - {_L3_TO_LABEL[l3]}" for l3 in sorted(want_l3))
    )

    raw, transform, raster_crs, nodata = read_raster_window_for_cams(
        corine_filepath, corine_band, cams_cells
    )

    # Pixel indices (1..44) we want to keep.
    wanted_indices = np.fromiter(
        (idx for idx, l3, _ in _CORINE_CLASSES if l3 in want_l3),
        dtype=np.int16,
    )
    raw_i = np.asarray(raw, dtype=np.int16)
    height, width = raw_i.shape
    inside = pixels_inside_cams_cells(height, width, transform, raster_crs, cams_cells)
    if nodata is not None:
        inside &= raw_i != int(nodata)

    if return_l3:
        out_l3 = np.zeros((height, width), dtype=np.int16)
        for l3 in sorted(want_l3):
            hit = (raw_i == _L3_TO_INDEX[int(l3)]) & inside
            out_l3[hit] = int(l3)
        log.info(
            "CORINE L3 labels: "
            + " ".join(f"{c}={int((out_l3 == c).sum())}" for c in sorted(want_l3))
        )
        out = out_l3
    else:
        mask = np.isin(raw_i, wanted_indices) & inside
        out = mask.astype(np.float32)
        log.info(f"CORINE: {int(out.sum())} pixels match selected land covers")

    if not need_cell_id:
        return out, transform, raster_crs, np.empty(0, dtype=np.int64)

    h, w = out.shape
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

    return out, transform, raster_crs, cell_id


def load_corine_weighted_l3(
    corine_filepath: Path,
    l3_weights: dict[int, float],
    corine_band: int,
    cams_cells: dict[int, dict[str, Any]],
    *,
    ref_height: int,
    ref_width: int,
    ref_transform: Any,
    ref_crs: Any,
) -> np.ndarray:
    """
    One CORINE window read; per-pixel sum of ``l3_weights[L3]`` on matching CLC classes (float32 0/1 masks).
    """
    from PROXY_V2.core.raster_helpers import warp_raster_to_grid

    if not l3_weights:
        return np.zeros((int(ref_height), int(ref_width)), dtype=np.float32)

    want_l3 = {int(k) for k in l3_weights}
    unknown = want_l3 - _L3_TO_LABEL.keys()
    if unknown:
        raise ValueError(f"Unknown CORINE L3 code(s): {sorted(unknown)}")

    log.info(
        "CORINE pasture classes:\n"
        + "\n".join(f"  {l3} w={l3_weights[l3]} — {_L3_TO_LABEL[l3]}" for l3 in sorted(want_l3))
    )

    raw, transform, raster_crs, nodata = read_raster_window_for_cams(
        corine_filepath, corine_band, cams_cells
    )
    raw_i = np.asarray(raw, dtype=np.int16)
    h, w = raw_i.shape
    inside = pixels_inside_cams_cells(h, w, transform, raster_crs, cams_cells)

    acc = np.zeros((h, w), dtype=np.float32)
    for l3, weight in l3_weights.items():
        if float(weight) == 0.0:
            continue
        idx = _L3_TO_INDEX[int(l3)]
        hit = (raw_i == idx) & inside
        if nodata is not None:
            hit &= raw_i != int(nodata)
        acc[hit] += np.float32(weight)
        log.debug(f"CORINE L3 {l3}: {int(hit.sum())} px weight={weight}")

    log.info(f"CORINE weighted pasture (native grid): sum={float(acc.sum()):.6g}")
    return warp_raster_to_grid(
        acc, transform, raster_crs, ref_height, ref_width, ref_transform, ref_crs, dest_init_nan=False,
    )


def load_corine_crop_groups(
    corine_filepath: Path,
    corine_band: int,
    corine_cfg: dict[str, Any],
    crop_groups: list[str],
    cams_cells: dict[int, dict[str, Any]],
    *,
    priority: list[str],
    ref_height: int,
    ref_width: int,
    ref_transform: Any,
    ref_crs: Any,
) -> np.ndarray:
    """
    Per-pixel crop-group id (1..len(crop_groups), 0 = none) on the reference grid.

    Reuses ``load_corine(..., return_l3=True)``; *priority* paints low-to-high (last wins).
    """
    from PROXY_V2.core.raster_helpers import warp_raster_to_grid
    from rasterio.enums import Resampling

    group_to_idx = {g: i + 1 for i, g in enumerate(crop_groups)}
    all_l3: list[int] = []
    for gname in crop_groups:
        key = f"{gname}_l3_codes"
        if key not in corine_cfg:
            raise ValueError(f"corine.{key} required")
        all_l3.extend(int(x) for x in corine_cfg[key])

    out_l3, transform, raster_crs, _ = load_corine(
        corine_filepath,
        list(set(all_l3)),
        int(corine_band),
        cams_cells,
        {},
        need_cell_id=False,
        return_l3=True,
    )
    gid_native = np.zeros(out_l3.shape, dtype=np.int8)
    for gname in reversed(priority):
        if gname not in group_to_idx:
            raise ValueError(f"unknown crop group {gname!r} in lucas_crop_group_priority")
        idx = group_to_idx[gname]
        for l3 in [int(x) for x in corine_cfg[f"{gname}_l3_codes"]]:
            gid_native[out_l3 == l3] = idx

    log.info(
        "CLC crop groups (native): "
        + " ".join(f"{g}={int((gid_native == group_to_idx[g]).sum())}" for g in crop_groups)
    )
    gid = warp_raster_to_grid(
        gid_native.astype(np.float32),
        transform,
        raster_crs,
        int(ref_height),
        int(ref_width),
        ref_transform,
        ref_crs,
        resampling=Resampling.nearest,
        dest_init_nan=False,
    ).astype(np.int8)
    log.info(
        "CLC crop groups (ref grid): "
        + " ".join(f"{g}={int((gid == group_to_idx[g]).sum())}" for g in crop_groups)
    )
    return gid


def _read_corine_band_clipped_to_metric_polygon(
    corine_filepath: Path,
    band: int,
    clip_polygon_metric: Any,
    metric_crs: str,
) -> tuple[np.ndarray, Any, Any, float | None]:
    """Read one CORINE band cropped to *clip_polygon_metric* (same metric domain as OSM aviation clip)."""
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_polygon_metric], crs=metric_crs)
    with rasterio.open(corine_filepath) as src:
        if not (1 <= band <= src.count):
            raise ValueError(f"band={band} out of range (file has {src.count} bands)")
        clip_re = clip_gdf.to_crs(src.crs)
        geom = mapping(clip_re.geometry.iloc[0])
        nd = src.nodata
        fill = float(nd) if nd is not None else 0.0
        try:
            arr, out_tr = rio_mask(
                src,
                [geom],
                crop=True,
                indexes=band,
                filled=True,
                nodata=fill,
            )
        except ValueError as exc:
            log.warning("CORINE mask clip produced no overlap: %s", exc)
            return np.zeros((0, 0), dtype=np.float64), None, src.crs, nd
        data = np.asarray(arr, dtype=np.float64)
        if data.ndim == 3:
            data = data[0]
        return data, out_tr, src.crs, nd


def _build_corine_airport_facilities_from_band(
    data: np.ndarray,
    transform: Any,
    raster_crs: Any,
    l3_codes: list[int],
    *,
    nodata: float | None,
    min_patch_area_m2: float,
    metric_crs: str,
) -> dict[str, dict[str, Any]]:
    """Vectorise CORINE airport pixels into patches; one facility per patch (metric CRS → WGS84 lon/lat)."""
    if data.size == 0 or transform is None:
        return {}

    wanted = corine_codes_matching(l3_codes, 1)
    want_arr = np.asarray(sorted(wanted), dtype=np.int64)
    work_i = np.round(np.asarray(data, dtype=np.float64)).astype(np.int64)
    airport = np.isin(work_i, want_arr)
    if nodata is not None:
        airport &= work_i != int(round(float(nodata)))
    if not np.any(airport):
        log.info("CORINE (clipped): no airport-class pixels in window")
        return {}

    out: dict[str, dict[str, Any]] = {}
    patch_i = 0
    for geom_dict, val in rio_features.shapes(
        airport.astype(np.uint8), mask=airport, transform=transform
    ):
        if int(val) == 0:
            continue
        poly_map = shp_shape(geom_dict)
        if poly_map.is_empty:
            continue
        g_m = gpd.GeoDataFrame(geometry=[poly_map], crs=raster_crs).to_crs(metric_crs)
        geom_m = g_m.geometry.iloc[0]
        area_m2 = float(geom_m.area)
        if area_m2 < float(min_patch_area_m2):
            continue
        pt_m = _geom_match_point_m3035(geom_m, None)
        ll = gpd.GeoDataFrame(geometry=[pt_m], crs=metric_crs).to_crs("EPSG:4326").geometry.iloc[0]
        l3 = int(l3_codes[0]) if len(l3_codes) == 1 else 124
        fid = f"corine:{l3}:{patch_i}"
        patch_i += 1
        out[fid] = {
            "lon": float(ll.x),
            "lat": float(ll.y),
            "name": f"CORINE L{l3}",
            "l3": l3,
            "area_m2": area_m2,
            "facility_id": fid,
        }

    log.info(f"CORINE airport patches in clip: {len(out)} (L3 {l3_codes})")
    return out


def get_corine_airport_facilities(
    corine_filepath: Path,
    band: int,
    l3_codes: list[int],
    clip_polygon_metric: Any,
    metric_crs: str,
    *,
    min_patch_area_m2: float,
) -> dict[str, dict[str, Any]]:
    """Clip CORINE to *clip_polygon_metric*, then one match point per connected L3 airport patch."""
    data, tr, rcrs, nd = _read_corine_band_clipped_to_metric_polygon(
        corine_filepath, band, clip_polygon_metric, metric_crs
    )
    return _build_corine_airport_facilities_from_band(
        data,
        tr,
        rcrs,
        l3_codes,
        nodata=nd,
        min_patch_area_m2=min_patch_area_m2,
        metric_crs=metric_crs,
    )
