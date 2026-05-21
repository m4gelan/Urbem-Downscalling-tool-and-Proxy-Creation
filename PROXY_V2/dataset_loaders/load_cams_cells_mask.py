from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.warp import transform_bounds

from PROXY_V2.core import log
from PROXY_V2.core.raster_helpers import pixel_centre_axes
from PROXY_V2.core.alias import cams_country_index_from_iso3, cams_pollutant_var


# ---------------------------------------------------------------------------
# Raster helpers (shared by load_population / load_corine)
# ---------------------------------------------------------------------------

def _union_bounds_wgs84(
    cams_cells: dict[int, dict[str, Any]],
) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) covering all CAMS cells, in WGS84."""
    if not cams_cells:
        raise ValueError("cams_cells is empty — cannot define a raster window")
    bounds = [c["cell_bounds_wgs84"] for c in cams_cells.values()]
    west  = min(b["west"]  for b in bounds)
    east  = max(b["east"]  for b in bounds)
    south = min(b["south"] for b in bounds)
    north = max(b["north"] for b in bounds)
    return west, south, east, north


def _union_bounds_raster_crs(
    cams_cells: dict[int, dict[str, Any]],
    dst_crs: Any,
) -> tuple[float, float, float, float]:
    """
    Union of CAMS cell bounds **in dst_crs**, each box transformed with densification.

    Tighter than transforming one WGS84 envelope when cells are separated: the lon/lat
    AABB can balloon in projected space along edges where no cell exists.
    """
    lefts: list[float] = []
    rights: list[float] = []
    bottoms: list[float] = []
    tops: list[float] = []
    for cell in cams_cells.values():
        b = cell["cell_bounds_wgs84"]
        w, s, e, n = b["west"], b["south"], b["east"], b["north"]
        l, btm, r, top = transform_bounds(
            "EPSG:4326", dst_crs, w, s, e, n, densify_pts=21
        )
        lefts.append(l)
        rights.append(r)
        bottoms.append(btm)
        tops.append(top)
    return min(lefts), min(bottoms), max(rights), max(tops)


def _centre_index_range(c: float, a: float, n: int, vmin: float, vmax_excl: float) -> tuple[int, int]:
    """Indices k in ``[0, n)`` with ``vmin <= c + (k + 0.5) * a < vmax_excl``; return ``k0, k1`` (``k1`` exclusive)."""
    if n <= 0 or abs(a) < 1e-30:
        return 0, 0
    eps = 1e-9
    if a > 0:
        k0 = int(np.ceil((vmin - c) / a - 0.5 - eps))
        k1 = int(np.floor((vmax_excl - c) / a - 0.5 + eps)) + 1
    else:
        k0 = int(np.ceil((vmax_excl - c) / a - 0.5 - eps))
        k1 = int(np.floor((vmin - c) / a - 0.5 + eps)) + 1
    k0 = max(0, min(n, k0))
    k1 = max(0, min(n, k1))
    if k1 < k0:
        return 0, 0
    return k0, k1


def read_raster_window_for_cams(
    raster_path: Path,
    band: int,
    cams_cells: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, rasterio.Affine, Any, float | None]:
    """
    Read only the GeoTIFF window covering all CAMS cell bounds.

    Returns
    -------
    (data, window_transform, crs, nodata)
    """
    # 1. Read the raster window: union of **per-cell** WGS84 bounds transformed into
    #    the raster CRS (tighter than one global WGS84 envelope → projected bounds).
    with rasterio.open(raster_path) as src:
        if not (1 <= band <= src.count):
            raise ValueError(
                f"band={band} out of range (file '{raster_path.name}' has {src.count} bands)"
            )

        if src.crs is None:
            w, s, e, n = _union_bounds_wgs84(cams_cells)
            left, bottom, right, top = transform_bounds(
                "EPSG:4326", "EPSG:4326", w, s, e, n, densify_pts=21
            )
        else:
            left, bottom, right, top = _union_bounds_raster_crs(cams_cells, src.crs)
        window = (
            rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
            .intersection(rasterio.windows.Window(0, 0, src.width, src.height))
            .round_lengths(op="ceil")
            .round_offsets(op="floor")
        )

        # 2. Read the raster data
        data = src.read(band, window=window)

        win_transform = src.window_transform(window)
        win_h, win_w = data.shape[0], data.shape[1]
        pixel_w = abs(src.transform.a)
        pixel_h = abs(src.transform.e)
        pixel_area_m2 = abs(src.transform.a * src.transform.e)
        total_area_km2 = (win_w * win_h * pixel_area_m2) / 1e6
        area_str = f"{total_area_km2:,.0f} km²"
        res_str = f"{pixel_w:.1f}×{pixel_h:.1f} m"
        log.info(
            f"Raster window {raster_path.name}: {win_w}×{win_h} px @ {res_str} "
            f"[CRS={src.crs}] → {area_str}"
        )
        return data, win_transform, src.crs, src.nodata


def pixels_inside_cams_cells(
    height: int,
    width: int,
    transform: rasterio.Affine,
    raster_crs: Any,
    cams_cells: dict[int, dict[str, Any]],
) -> np.ndarray:
    """True where the pixel centre falls inside any CAMS cell (window grid)."""
    pixel_x, pixel_y = pixel_centre_axes(transform, height, width)
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    axis_aligned = abs(b) < 1e-10 and abs(d) < 1e-10

    # If raster is not WGS84, reproject the cell bounds (cheap: a few per cell)
    # rather than every pixel.
    to_raster = None
    if raster_crs is not None and raster_crs.to_epsg() != 4326:
        to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

    inside = np.zeros((height, width), dtype=bool)
    for cell in cams_cells.values():
        wb = cell["cell_bounds_wgs84"]
        lon_w, lon_e, lat_s, lat_n = wb["west"], wb["east"], wb["south"], wb["north"]
        if to_raster is not None:
            # All four WGS84 corners — a lat/lon box is not axis-aligned in EPSG:3035.
            xs, ys = to_raster.transform(
                [lon_w, lon_w, lon_e, lon_e],
                [lat_s, lat_n, lat_s, lat_n],
            )
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
            xmin, xmax = lon_w, lon_e
            ymin, ymax = lat_s, lat_n

        if axis_aligned:
            j0, j1 = _centre_index_range(c, a, width, xmin, xmax)
            i0, i1 = _centre_index_range(f, e, height, ymin, ymax)
            if j0 >= j1 or i0 >= i1:
                continue
            px = pixel_x[j0:j1]
            py = pixel_y[i0:i1]
            sub = (px[None, :] >= xmin) & (px[None, :] < xmax) & (py[:, None] >= ymin) & (
                py[:, None] < ymax
            )
            inside[i0:i1, j0:j1] |= sub
        else:
            col_hit = (pixel_x >= xmin) & (pixel_x < xmax)
            row_hit = (pixel_y >= ymin) & (pixel_y < ymax)
            if col_hit.any() and row_hit.any():
                inside |= row_hit[:, None] & col_hit[None, :]
    return inside


# ---------------------------------------------------------------------------
# CAMS area-source aggregation
# ---------------------------------------------------------------------------

def load_cams_cells_mask(
    cams_nc: Path,
    *,
    year: int,
    country_iso3: str,
    emission_category_indices: list[int],
    source_type_indices: list[int],
    pollutants: list[str],
    crs: str,
    resolution_m: float,
    pad_m: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """
    Aggregate CAMS area-source rows for one country into unique grid cells.

    Each row's (longitude_index, latitude_index) maps to a cell; emissions are
    summed per pollutant over all rows in the same cell.

    Parameters ``crs``, ``resolution_m``, ``pad_m`` are accepted for signature
    parity with the raster pipeline but unused here.

    Returns ``{cell_id: row}`` with ``cell_id = lat_idx * n_lon + lon_idx`` (0-based).
    """
    _ = (crs, resolution_m, pad_m)

    iso3 = country_iso3.strip().upper()
    ec_filter = np.asarray(emission_category_indices, dtype=np.int64)
    st_filter = np.asarray(source_type_indices, dtype=np.int64)
    pollutant_labels = [p.strip() for p in pollutants if p and p.strip()]
    if not pollutant_labels:
        raise ValueError("`pollutants` must contain at least one label")

    with xr.open_dataset(cams_nc, engine="netcdf4") as ds:
        country_idx = cams_country_index_from_iso3(ds, iso3)

        lon_src       = np.asarray(ds["longitude_source"].values,        dtype=np.float64).ravel()
        lat_src       = np.asarray(ds["latitude_source"].values,         dtype=np.float64).ravel()
        src_type      = np.asarray(ds["source_type_index"].values,       dtype=np.int64  ).ravel()
        emis_cat      = np.asarray(ds["emission_category_index"].values, dtype=np.int64  ).ravel()
        country_index = np.asarray(ds["country_index"].values,           dtype=np.int64  ).ravel()
        lon_idx       = np.asarray(ds["longitude_index"].values,         dtype=np.int64  ).ravel()
        lat_idx       = np.asarray(ds["latitude_index"].values,          dtype=np.int64  ).ravel()

        nlon = int(ds.sizes["longitude"])
        nlat = int(ds.sizes["latitude"])
        lon_bounds = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
        lat_bounds = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)

        # Detect 1-based indexing (some CAMS files do that) and shift.
        if lon_idx.size and (lon_idx.max() >= nlon or lat_idx.max() >= nlat):
            log.debug("CAMS indices look 1-based; shifting to 0-based")
            lon_idx = lon_idx - 1
            lat_idx = lat_idx - 1
        np.clip(lon_idx, 0, nlon - 1, out=lon_idx)
        np.clip(lat_idx, 0, nlat - 1, out=lat_idx)

        # (n_sources, n_pollutants) — one allocation, NaN-safe.
        pollutant_matrix = np.column_stack([
            np.nan_to_num(
                np.asarray(ds[cams_pollutant_var(lab)].values, dtype=np.float64).ravel(),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            for lab in pollutant_labels
        ])

    # --- Source filter ------------------------------------------------------
    mask = (
        np.isfinite(lon_src)
        & np.isfinite(lat_src)
        & np.isin(emis_cat, ec_filter)
        & np.isin(src_type, st_filter)
        & (country_index == int(country_idx))
        & (pollutant_matrix.max(axis=1) > 0.0)
    )
    sel = np.flatnonzero(mask)
    if sel.size == 0:
        log.warning(
            f"No CAMS sources after filters "
            f"(country={iso3}, emis_cat={ec_filter.tolist()}, "
            f"source_type={st_filter.tolist()})."
        )
        return {}, {
            "lon_bounds": np.asarray(lon_bounds, dtype=np.float64),
            "lat_bounds": np.asarray(lat_bounds, dtype=np.float64),
            "n_longitude": nlon,
            "n_latitude": nlat,
        }

    # --- Aggregate per cell with bincount (vectorised) ----------------------
    cell_ids = lat_idx[sel] * nlon + lon_idx[sel]
    unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
    n_cells = unique_cells.size

    sums = np.empty((n_cells, len(pollutant_labels)), dtype=np.float64)
    for j in range(len(pollutant_labels)):
        sums[:, j] = np.bincount(
            inverse, weights=pollutant_matrix[sel, j], minlength=n_cells
        )

    # --- Build output dict --------------------------------------------------
    out: dict[int, dict[str, Any]] = {}
    for k, cell_id in enumerate(unique_cells.tolist()):
        lo = int(cell_id %  nlon)
        la = int(cell_id // nlon)
        west,  east  = float(lon_bounds[lo, 0]), float(lon_bounds[lo, 1])
        south, north = float(lat_bounds[la, 0]), float(lat_bounds[la, 1])
        out[int(cell_id)] = {
            "pollutants_within_cell": {
                lab: float(sums[k, j]) for j, lab in enumerate(pollutant_labels)
            },
            "longitude":  0.5 * (west + east),
            "latitude":   0.5 * (south + north),
            "width_deg":  east  - west,
            "height_deg": north - south,
            "cell_bounds_wgs84": {
                "west": west, "south": south, "east": east, "north": north,
            },
            "longitude_index": lo,
            "latitude_index":  la,
            "n_longitude": nlon,
            "n_latitude":  nlat,
            "year": int(year),
            "country_iso3": iso3,
        }

    cell_width_km = (lon_bounds[0,1] - lon_bounds[0,0]) * 111  # rough km at equator
    cell_height_km = (lat_bounds[0,1] - lat_bounds[0,0]) * 111
    pixel_area_km2 = cell_width_km * cell_height_km
    log.info(
        f"CAMS area grid cells (country={iso3}): {len(out)} unique cells "
        f"grid {nlon}x{nlat} pix, cell size ≈ {cell_width_km:.2f}x{cell_height_km:.2f} km, "
        f"total grid area ≈ {nlon*nlat*pixel_area_km2} km^2"
    )
    
    grid = {
        "lon_bounds": np.asarray(lon_bounds, dtype=np.float64),
        "lat_bounds": np.asarray(lat_bounds, dtype=np.float64),
        "n_longitude": nlon,
        "n_latitude": nlat,
    }
    return out, grid
