"""
Per-CAMS-cell area weights on CORINE clip pixels (blend or legacy CORINE × population^β).

**Workflow (see also** ``A_PublicPower/README.md`` **):**

1. **Iterate** each CAMS source that is true in the area mask: get lon/lat and cell
   bounds in WGS84.
2. **Clip** CORINE to the CAMS cell polygon (rasterio mask), in native CORINE CRS.
3. **Resample** the population raster to that clip's grid and shape (bilinear or
   nearest as configured).
4. **Form** a non-negative per-pixel weight tensor, **normalize to sum 1** on the
   clip (a *share* tensor).
5. **Accumulate** each clip pixel's share onto the **reference** output grid:
   for each non-zero share, map the clip pixel **centre** to ref row/col and
   ``np.add.at`` (so multiple CAMS cells can contribute to the same ref pixel).

**Weight models** (``area_proxy.weight_model``):

* ``eligibility_pop_blend`` — default; blends CORINE class eligibility with
  cell-local min–max population; see :func:`_share_tensor_eligibility_pop_blend`.
* ``corine_pop_product`` — legacy; eligibility × population^β; see
  :func:`_share_tensor_corine_pop_product`.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd
    import rasterio.io
    import xarray as xr


def cell_overlaps_domain_bbox(
    west: float,
    south: float,
    east: float,
    north: float,
    bbox_wgs84: tuple[float, float, float, float],
) -> bool:
    bw, bs, be, bn = bbox_wgs84
    if west > east:
        west, east = east, west
    if south > north:
        south, north = north, south
    return not (east < bw or west > be or north < bs or south > bn)


def _cams_lonlat_indices(ds: "xr.Dataset", m_area: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Pack CAMS 1D **area** source indices for downstream iteration.

    The NetCDF may use 0- or 1-based ``*_index`` columns; a defensive shift keeps
    bounds access in range. Returns the flat indices where ``m_area`` is true so
    the main loop only visits relevant sources.
    """
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
    lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
    nlon = int(lon_b.shape[0])
    nlat = int(lat_b.shape[0])
    lon_idx_raw = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
    lat_idx_raw = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
    if lon_idx_raw.max() >= nlon or lat_idx_raw.max() >= nlat:
        lon_idx_raw = np.maximum(0, lon_idx_raw - 1)
        lat_idx_raw = np.maximum(0, lat_idx_raw - 1)
    lon_ii = np.clip(lon_idx_raw, 0, nlon - 1)
    lat_ii = np.clip(lat_idx_raw, 0, nlat - 1)
    return lon, lat, lon_b, lat_b, lon_ii, lat_ii, np.flatnonzero(m_area)


def _clip_corine_masked(
    corine_src: "rasterio.io.DatasetReader",
    geom_native: object,
    corine_band: int,
) -> tuple[np.ndarray, object] | None:
    """Crop CORINE to the CAMS cell geometry; return array + transform or ``None`` if empty."""
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import mapping

    try:
        corine_win, corine_tr = rio_mask(
            corine_src,
            [mapping(geom_native)],
            crop=True,
            indexes=int(corine_band),
        )
    except ValueError:
        return None
    cw = corine_win.astype(np.float64)
    corine_arr = cw[0] if cw.ndim == 3 else cw
    nodata = corine_src.nodata
    if nodata is not None:
        corine_arr = np.where(corine_arr == float(nodata), np.nan, corine_arr)
    if corine_arr.size < 1:
        return None
    return corine_arr, corine_tr


def _warp_population_to_corine(
    pop_src: "rasterio.io.DatasetReader",
    corine_crs: object,
    corine_tr: object,
    height: int,
    width: int,
    resampling: Literal["bilinear", "nearest"],
) -> np.ndarray:
    """
    Reproject a window of the **global** population raster to match the CORINE clip
    grid (same height/width, CRS, and transform as the clip). NaN where no data.
    """
    from rasterio.transform import array_bounds
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    pop_dst = np.full((height, width), np.nan, dtype=np.float64)
    left, bottom, right, top = array_bounds(height, width, corine_tr)
    l0, b0, r0, t0 = transform_bounds(
        corine_crs, pop_src.crs, left, bottom, right, top, densify_pts=21
    )
    win_pop = from_bounds(l0, b0, r0, t0, transform=pop_src.transform).intersection(
        Window(0, 0, pop_src.width, pop_src.height)
    )
    rs = (
        Resampling.bilinear if resampling == "bilinear" else Resampling.nearest
    )
    if win_pop.width < 1 or win_pop.height < 1:
        pop_dst.fill(0.0)
        return pop_dst

    pop_arr = pop_src.read(1, window=win_pop).astype(np.float64)
    pop_tr = pop_src.window_transform(win_pop)
    pop_nodata = pop_src.nodata
    if pop_nodata is not None:
        pop_arr = np.where(pop_arr == float(pop_nodata), np.nan, pop_arr)
    reproject(
        source=pop_arr,
        destination=pop_dst,
        src_transform=pop_tr,
        src_crs=pop_src.crs,
        dst_transform=corine_tr,
        dst_crs=corine_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=rs,
    )
    return pop_dst


def _pop_01_within_cell(pop_work: np.ndarray, ok_data: np.ndarray) -> np.ndarray:
    """Min–max population to [0, 1] over finite-CORINE pixels in this CAMS cell clip."""
    pop_01 = np.zeros_like(pop_work, dtype=np.float64)
    if not np.any(ok_data):
        return pop_01
    p = pop_work[ok_data]
    pmin = float(np.min(p))
    pmax = float(np.max(p))
    if pmax > pmin:
        pop_01[ok_data] = (pop_work[ok_data] - pmin) / (pmax - pmin)
    else:
        pop_01[ok_data] = 1.0 if pmax > 0.0 else 0.0
    return np.clip(pop_01, 0.0, 1.0)


def _share_tensor_corine_pop_product(
    corine_arr: np.ndarray,
    pop_dst: np.ndarray,
    code_set: frozenset[int],
    pop_exp: float,
    floor: float,
    fallback_if_no_corine: Literal["pop_in_cell", "skip"],
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Legacy: eligibility x pop^exponent, with fallbacks when no eligible CORINE."""
    ok_data = np.isfinite(corine_arr)
    rint = np.zeros_like(corine_arr, dtype=np.int32)
    rint[ok_data] = np.rint(corine_arr[ok_data]).astype(np.int32)
    corine_hit = ok_data & np.isin(rint, list(code_set))

    pop_safe = np.maximum(np.nan_to_num(pop_dst, nan=0.0), floor)
    w_pix = np.where(corine_hit, np.power(pop_safe, pop_exp), 0.0)
    s = float(np.sum(w_pix))
    basis = "corine_eligible_pop"

    if s <= 0:
        if fallback_if_no_corine == "skip":
            return None
        w_pix = np.where(ok_data, np.power(pop_safe, pop_exp), 0.0)
        s = float(np.sum(w_pix))
        basis = "pop_full_cell"
        if s <= 0:
            w_pix = np.where(ok_data, 1.0, 0.0)
            s = float(np.sum(w_pix))
            basis = "uniform_cell"
            if s <= 0:
                return None

    share = (w_pix / s).astype(np.float64, copy=False)
    return share, w_pix.astype(np.float64, copy=False), basis


def _share_tensor_eligibility_pop_blend(
    corine_arr: np.ndarray,
    pop_dst: np.ndarray,
    code_set: frozenset[int],
    floor: float,
    fallback_if_no_corine: Literal["pop_in_cell", "skip"],
    *,
    blend_eligibility_coef: float,
    blend_population_coef: float,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Per-pixel score on the CORINE clip (normalized to sum 1 in the cell):

        pop_01 = min–max(pop) in [0, 1] within the cell (finite CORINE pixels only)
        score = a * eligibility^(1 + pop_01) + b * pop_01

    ``eligibility`` is 1 on target CORINE classes else 0. Population always contributes via ``b * pop_01``.
    """
    ok_data = np.isfinite(corine_arr)
    if not np.any(ok_data):
        return None

    rint = np.zeros_like(corine_arr, dtype=np.int32)
    rint[ok_data] = np.rint(corine_arr[ok_data]).astype(np.int32)
    corine_hit = ok_data & np.isin(rint, list(code_set))

    pop_work = np.maximum(np.nan_to_num(pop_dst, nan=0.0), floor)
    pop_01 = _pop_01_within_cell(pop_work, ok_data)

    elig = corine_hit.astype(np.float64)
    exp_elig = np.zeros_like(corine_arr, dtype=np.float64)
    exp_elig[ok_data] = np.power(elig[ok_data], 1.0 + pop_01[ok_data])

    w_pix = np.zeros_like(corine_arr, dtype=np.float64)
    w_pix[ok_data] = (
        float(blend_eligibility_coef) * exp_elig[ok_data]
        + float(blend_population_coef) * pop_01[ok_data]
    )
    s = float(np.sum(w_pix))
    basis = "eligibility_pop_blend"

    if s <= 0:
        if fallback_if_no_corine == "skip":
            return None
        w_pix = np.where(ok_data, 1.0, 0.0)
        s = float(np.sum(w_pix))
        basis = "uniform_cell"
        if s <= 0:
            return None

    share = (w_pix / s).astype(np.float64, copy=False)
    return share, w_pix.astype(np.float64, copy=False), basis


def _share_tensor_for_cell(
    corine_arr: np.ndarray,
    pop_dst: np.ndarray,
    code_set: frozenset[int],
    pop_exp: float,
    floor: float,
    fallback_if_no_corine: Literal["pop_in_cell", "skip"],
    *,
    weight_model: str = "eligibility_pop_blend",
    blend_eligibility_coef: float = 0.7,
    blend_population_coef: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Return (share, weight_raw, basis) on CORINE clip grid, or None if cell is skipped."""
    wm = str(weight_model).strip().lower().replace("-", "_")
    if wm in ("corine_pop_product", "legacy"):
        return _share_tensor_corine_pop_product(
            corine_arr,
            pop_dst,
            code_set,
            pop_exp,
            floor,
            fallback_if_no_corine,
        )
    if wm in ("eligibility_pop_blend", "blend"):
        return _share_tensor_eligibility_pop_blend(
            corine_arr,
            pop_dst,
            code_set,
            floor,
            fallback_if_no_corine,
            blend_eligibility_coef=blend_eligibility_coef,
            blend_population_coef=blend_population_coef,
        )
    raise ValueError(
        f"Unknown area_proxy weight_model {weight_model!r}; "
        "use 'eligibility_pop_blend' or 'corine_pop_product'."
    )


def _burn_share_to_ref(
    acc: np.ndarray,
    ref_tr: object,
    corine_tr: object,
    share: np.ndarray,
) -> int:
    """
    Distribute a CAMS cell's **share** image onto the reference grid.

    For each non-zero share pixel, use the **centre** of the CORINE pixel in CRS,
    reproject to ref indices, and accumulate (so overlapping CAMS cells can stack).
    """
    from rasterio.transform import rowcol, xy

    ys, xs = np.nonzero(share > 0)
    if ys.size == 0:
        return 0
    sh = share[ys, xs].astype(np.float64, copy=False)
    xc, yc = xy(corine_tr, ys, xs, offset="center")
    ri, ci = rowcol(ref_tr, xc, yc)
    ri = np.asarray(ri, dtype=np.intp)
    ci = np.asarray(ci, dtype=np.intp)
    m = (ri >= 0) & (ri < acc.shape[0]) & (ci >= 0) & (ci < acc.shape[1])
    if not np.any(m):
        return 0
    np.add.at(acc, (ri[m], ci[m]), sh[m])
    return int(np.count_nonzero(m))


def _rows_for_cams_cell(
    *,
    i: int,
    lon_i: float,
    lat_i: float,
    west: float,
    south: float,
    east: float,
    north: float,
    corine_arr: np.ndarray,
    corine_tr: object,
    corine_crs: object,
    pop_dst: np.ndarray,
    code_set: frozenset[int],
    pop_exp: float,
    floor: float,
    fallback_if_no_corine: Literal["pop_in_cell", "skip"],
    weight_model: str = "eligibility_pop_blend",
    blend_eligibility_coef: float = 0.7,
    blend_population_coef: float = 0.3,
) -> list[dict[str, Any]]:
    import geopandas as gpd
    from rasterio.transform import xy as transform_xy
    from shapely.geometry import Polygon

    packed = _share_tensor_for_cell(
        corine_arr,
        pop_dst,
        code_set,
        pop_exp,
        floor,
        fallback_if_no_corine,
        weight_model=weight_model,
        blend_eligibility_coef=blend_eligibility_coef,
        blend_population_coef=blend_population_coef,
    )
    if packed is None:
        return []
    share, w_pix, basis = packed

    ok_data = np.isfinite(corine_arr)
    rint = np.zeros_like(corine_arr, dtype=np.int32)
    rint[ok_data] = np.rint(corine_arr[ok_data]).astype(np.int32)
    corine_hit = ok_data & np.isin(rint, list(code_set))
    ys, xs = np.nonzero(share > 0)
    rows: list[dict[str, Any]] = []
    for row, col in zip(ys, xs):
        sh = float(share[row, col])
        wr = float(w_pix[row, col])
        if sh <= 0:
            continue
        corners = []
        for r_off, c_off in ((0, 0), (0, 1), (1, 1), (1, 0)):
            x, y = transform_xy(corine_tr, row + r_off, col + c_off, offset="ul")
            corners.append((float(x), float(y)))
        poly_native = Polygon(corners)
        if not poly_native.is_valid:
            poly_native = poly_native.buffer(0)
        geom = (
            gpd.GeoDataFrame(geometry=[poly_native], crs=corine_crs)
            .to_crs(4326)
            .geometry.iloc[0]
        )
        cv = int(rint[row, col]) if ok_data[row, col] else -9999
        pop_v = (
            float(pop_dst[row, col])
            if np.isfinite(pop_dst[row, col])
            else 0.0
        )
        rows.append(
            {
                "cams_source_index": int(i),
                "weight_raw": wr,
                "weight_share": sh,
                "population_proxy": pop_v,
                "corine_value": cv,
                "in_corine_target": bool(corine_hit[row, col]),
                "weight_basis": basis,
                "lon_cams_centre": float(lon_i),
                "lat_cams_centre": float(lat_i),
                "cell_west": west,
                "cell_south": south,
                "cell_east": east,
                "cell_north": north,
                "geometry": geom,
            }
        )
    return rows


def iter_masked_corine_cells(
    ds: "xr.Dataset",
    area_mask: np.ndarray,
    *,
    corine_path: Path,
    population_path: Path,
    corine_band: int,
    population_resampling: Literal["bilinear", "nearest"],
    domain_bbox_wgs84: tuple[float, float, float, float] | None,
    show_progress: bool,
) -> Iterator[
    tuple[
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        np.ndarray,
        object,
        object,
        np.ndarray,
    ]
]:
    """
    For each true entry in ``area_mask``, build CORINE + population on the CAMS **cell** footprint.

    Yields ``(i, lon, lat, west, south, east, north, corine_arr, corine_tr,
    corine_crs, pop_dst)`` so downstream code can form weights on ``corine_arr``
    and burn them to the reference using ``corine_tr``.
    """
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box

    lon, lat, lon_b, lat_b, lon_ii, lat_ii, area_idx = _cams_lonlat_indices(
        ds, area_mask
    )
    _iter = area_idx
    if show_progress:
        try:
            from tqdm import tqdm

            _iter = tqdm(
                area_idx,
                desc="CAMS area cells (CORINE / population)",
                unit="cell",
                total=int(area_idx.size),
                file=sys.stderr,
                disable=not sys.stderr.isatty(),
                mininterval=0.5,
            )
        except ImportError:
            pass

    with rasterio.open(corine_path) as corine_src, rasterio.open(
        population_path
    ) as pop_src:
        if corine_band < 1 or corine_band > int(corine_src.count):
            raise ValueError(f"corine_band {corine_band} invalid for {corine_path}")
        corine_crs = corine_src.crs
        if corine_crs is None:
            raise ValueError(f"CORINE raster has no CRS: {corine_path}")
        if pop_src.crs is None:
            raise ValueError(f"Population raster has no CRS: {population_path}")

        for i in _iter:
            # CAMS cell axis-aligned bounds in WGS84 (same convention as the NetCDF).
            li, ji = int(lon_ii[i]), int(lat_ii[i])
            west, east = float(lon_b[li, 0]), float(lon_b[li, 1])
            south, north = float(lat_b[ji, 0]), float(lat_b[ji, 1])
            if south > north:
                south, north = north, south
            if west > east:
                west, east = east, west

            if domain_bbox_wgs84 is not None and not cell_overlaps_domain_bbox(
                west, south, east, north, domain_bbox_wgs84
            ):
                continue

            poly4326 = gpd.GeoDataFrame(
                geometry=[box(west, south, east, north)], crs="EPSG:4326"
            )
            geom_native = poly4326.to_crs(corine_crs).geometry.iloc[0]
            if geom_native.is_empty:
                continue

            clip = _clip_corine_masked(corine_src, geom_native, corine_band)
            if clip is None:
                continue
            corine_arr, corine_tr = clip
            h, w = corine_arr.shape
            if h < 1 or w < 1:
                continue

            pop_dst = _warp_population_to_corine(
                pop_src,
                corine_crs,
                corine_tr,
                h,
                w,
                population_resampling,
            )
            yield (
                int(i),
                float(lon[i]),
                float(lat[i]),
                west,
                south,
                east,
                north,
                corine_arr,
                corine_tr,
                corine_crs,
                pop_dst,
            )


def burn_corine_population_weights_to_ref(
    ds: "xr.Dataset",
    area_mask: np.ndarray,
    ref_profile: dict[str, Any],
    *,
    corine_path: Path,
    population_path: Path,
    corine_codes: tuple[int, ...] = (121, 3),
    corine_band: int = 1,
    pop_exponent: float = 1.0,
    pop_floor: float = 0.0,
    population_resampling: Literal["bilinear", "nearest"] = "bilinear",
    fallback_if_no_corine: Literal["pop_in_cell", "skip"] = "pop_in_cell",
    domain_bbox_wgs84: tuple[float, float, float, float] | None = None,
    show_progress: bool = False,
    weight_model: str = "eligibility_pop_blend",
    blend_eligibility_coef: float = 0.7,
    blend_population_coef: float = 0.3,
) -> tuple[np.ndarray, int]:
    """
    Return the reference-grid accumulator (``float32``) and a diagnostic pixel count.

    The second value counts CORINE-clip **pixels** with positive share (summed
    across CAMS cells), not the number of CAMS cells. ``acc`` is **not** normalized
    to sum 1 globally; that may be done in a later pipeline stage if required.
    """
    code_set = frozenset(int(c) for c in corine_codes)
    pop_exp = float(pop_exponent)
    floor = max(0.0, float(pop_floor))
    wm = str(weight_model).strip()
    a_blend = float(blend_eligibility_coef)
    b_blend = float(blend_population_coef)
    rh = int(ref_profile["height"])
    rw = int(ref_profile["width"])
    ref_tr = ref_profile["transform"]
    acc = np.zeros((rh, rw), dtype=np.float64)
    n_contributing_pixels = 0

    for (
        _i,
        _lon_i,
        _lat_i,
        _w,
        _s,
        _e,
        _n,
        corine_arr,
        corine_tr,
        _corine_crs,
        pop_dst,
    ) in iter_masked_corine_cells(
        ds,
        area_mask,
        corine_path=corine_path,
        population_path=population_path,
        corine_band=corine_band,
        population_resampling=population_resampling,
        domain_bbox_wgs84=domain_bbox_wgs84,
        show_progress=show_progress,
    ):
        packed = _share_tensor_for_cell(
            corine_arr,
            pop_dst,
            code_set,
            pop_exp,
            floor,
            fallback_if_no_corine,
            weight_model=wm,
            blend_eligibility_coef=a_blend,
            blend_population_coef=b_blend,
        )
        if packed is None:
            continue
        share, _w_pix, _basis = packed
        n_contributing_pixels += int(np.count_nonzero(share > 0))
        _burn_share_to_ref(acc, ref_tr, corine_tr, share)

    return acc.astype(np.float32), n_contributing_pixels


