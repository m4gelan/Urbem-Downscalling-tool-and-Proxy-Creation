from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import geopandas as gpd
import rasterio
import xarray as xr
from pyproj import Transformer
from rasterio.warp import transform_bounds

from proxy.core import log
from proxy.core.raster_helpers import pixel_centre_axes
from proxy.core.alias import cams_country_index_from_iso3, cams_pollutant_var


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
        log.debug(f"Raster window {raster_path.name}")
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
    mass_source_type_indices: list[int] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """
    Build CAMS cell mask for proxy rasters.

    ``source_type_indices`` defines which cells belong to the sector (footprint).
    ``mass_source_type_indices`` (default: same as footprint) sets which rows
    contribute to ``pollutants_within_cell`` (area downscaling / alpha mass).
    """
    _ = (crs, resolution_m, pad_m)

    iso3 = country_iso3.strip().upper()
    ec_filter = np.asarray(emission_category_indices, dtype=np.int64)
    st_footprint = np.asarray(source_type_indices, dtype=np.int64)
    st_mass = np.asarray(
        mass_source_type_indices if mass_source_type_indices is not None else source_type_indices,
        dtype=np.int64,
    )
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
    footprint_mask = (
        np.isfinite(lon_src)
        & np.isfinite(lat_src)
        & np.isin(emis_cat, ec_filter)
        & np.isin(src_type, st_footprint)
        & (country_index == int(country_idx))
        & (pollutant_matrix.max(axis=1) > 0.0)
    )
    fp_sel = np.flatnonzero(footprint_mask)
    if fp_sel.size == 0:
        log.warning(
            f"No CAMS sources after filters "
            f"(country={iso3}, emis_cat={ec_filter.tolist()}, "
            f"source_type={st_footprint.tolist()})."
        )
        return {}, {
            "lon_bounds": np.asarray(lon_bounds, dtype=np.float64),
            "lat_bounds": np.asarray(lat_bounds, dtype=np.float64),
            "n_longitude": nlon,
            "n_latitude": nlat,
        }

    footprint_cells = np.unique(lat_idx[fp_sel] * nlon + lon_idx[fp_sel])

    mass_mask = (
        np.isfinite(lon_src)
        & np.isfinite(lat_src)
        & np.isin(emis_cat, ec_filter)
        & np.isin(src_type, st_mass)
        & (country_index == int(country_idx))
    )
    mass_sel = np.flatnonzero(mass_mask)
    if mass_sel.size == 0:
        unique_cells = footprint_cells
        sums = np.zeros((unique_cells.size, len(pollutant_labels)), dtype=np.float64)
    else:
        cell_ids = lat_idx[mass_sel] * nlon + lon_idx[mass_sel]
        in_footprint = np.isin(cell_ids, footprint_cells)
        if not np.any(in_footprint):
            unique_cells = footprint_cells
            sums = np.zeros((unique_cells.size, len(pollutant_labels)), dtype=np.float64)
        else:
            mass_sel = mass_sel[in_footprint]
            cell_ids = cell_ids[in_footprint]
            unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
            n_cells = unique_cells.size
            sums = np.empty((n_cells, len(pollutant_labels)), dtype=np.float64)
            for j in range(len(pollutant_labels)):
                sums[:, j] = np.bincount(
                    inverse, weights=pollutant_matrix[mass_sel, j], minlength=n_cells
                )
            missing = np.setdiff1d(footprint_cells, unique_cells)
            if missing.size:
                unique_cells = np.concatenate([unique_cells, missing])
                pad = np.zeros((missing.size, len(pollutant_labels)), dtype=np.float64)
                sums = np.vstack([sums, pad])

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

    cell_width_km = (lon_bounds[0, 1] - lon_bounds[0, 0]) * 111
    cell_height_km = (lat_bounds[0, 1] - lat_bounds[0, 0]) * 111
    covered_km2 = 0.0
    for cell in out.values():
        b = cell["cell_bounds_wgs84"]
        lat_mid = 0.5 * (b["south"] + b["north"])
        w_km = (b["east"] - b["west"]) * 111.0 * np.cos(np.radians(lat_mid))
        h_km = (b["north"] - b["south"]) * 111.0
        covered_km2 += w_km * h_km
    log.info(
        f"CAMS area grid cells (country={iso3}): {len(out)} unique cells "
        f"grid {nlon}x{nlat}, cell size ~ {cell_width_km:.2f}x{cell_height_km:.2f} km, "
        f"covered area ~ {covered_km2:,.0f} km2"
    )
    
    grid = {
        "lon_bounds": np.asarray(lon_bounds, dtype=np.float64),
        "lat_bounds": np.asarray(lat_bounds, dtype=np.float64),
        "n_longitude": nlon,
        "n_latitude": nlat,
    }
    return out, grid


def _cams_cell_centroids_wgs84(
    lon_bounds: np.ndarray,
    lat_bounds: np.ndarray,
    nlon: int,
    nlat: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat cell_id, lon, lat for every CAMS grid cell centre."""
    lo = np.arange(nlon, dtype=np.int64)
    la = np.arange(nlat, dtype=np.int64)
    lo2, la2 = np.meshgrid(lo, la, indexing="xy")
    cell_ids = (la2 * nlon + lo2).ravel()
    lon = (lon_bounds[lo2, 0] + lon_bounds[lo2, 1]) * 0.5
    lat = (lat_bounds[la2, 0] + lat_bounds[la2, 1]) * 0.5
    return cell_ids.ravel(), lon.ravel(), lat.ravel()


def _cells_centroid_in_maritime_domain(
    lon_bounds: np.ndarray,
    lat_bounds: np.ndarray,
    nlon: int,
    nlat: int,
    maritime_wgs84,
) -> np.ndarray:
    cell_ids, lons, lats = _cams_cell_centroids_wgs84(lon_bounds, lat_bounds, nlon, nlat)
    gdf = gpd.GeoDataFrame(
        {"cell_id": cell_ids},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326",
    )
    maritime_gdf = gpd.GeoDataFrame(geometry=[maritime_wgs84], crs="EPSG:4326")
    joined = gpd.sjoin(gdf, maritime_gdf, predicate="within", how="inner")
    return joined["cell_id"].to_numpy(dtype=np.int64)


def load_cams_shipping_cells_mask(
    cams_nc: Path,
    *,
    year: int,
    country_iso3: str,
    country_profile: dict[str, str],
    nuts_path: Path,
    maritime_buffer_m: float,
    maritime_metric_crs: str,
    emission_category_indices: list[int],
    source_type_indices: list[int],
    pollutants: list[str],
    crs: str,
    resolution_m: float,
    pad_m: float,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """
    G_Shipping footprint: country-attributed cells union maritime-buffer cells.
    Cell mass sums all sector sources in the cell (any CAMS country/domain).
    """
    from proxy.dataset_loaders.load_nuts2 import load_nuts0_country_polygon

    _ = (crs, resolution_m, pad_m)
    iso3 = country_iso3.strip().upper()
    ec_filter = np.asarray(emission_category_indices, dtype=np.int64)
    st_filter = np.asarray(source_type_indices, dtype=np.int64)
    pollutant_labels = [p.strip() for p in pollutants if p and p.strip()]
    if not pollutant_labels:
        raise ValueError("`pollutants` must contain at least one label")

    country_outline = load_nuts0_country_polygon(nuts_path, country_profile)
    maritime_geom = (
        country_outline.to_crs(maritime_metric_crs)
        .geometry.iloc[0]
        .buffer(float(maritime_buffer_m))
    )
    maritime_wgs84 = (
        gpd.GeoDataFrame(geometry=[maritime_geom], crs=maritime_metric_crs)
        .to_crs("EPSG:4326")
        .geometry.iloc[0]
    )
    log.info(
        f"G_Shipping maritime domain: NUTS0 + {float(maritime_buffer_m):g} m buffer "
        f"({maritime_metric_crs})"
    )

    with xr.open_dataset(cams_nc, engine="netcdf4") as ds:
        country_idx = cams_country_index_from_iso3(ds, iso3)
        lon_src = np.asarray(ds["longitude_source"].values, dtype=np.float64).ravel()
        lat_src = np.asarray(ds["latitude_source"].values, dtype=np.float64).ravel()
        src_type = np.asarray(ds["source_type_index"].values, dtype=np.int64).ravel()
        emis_cat = np.asarray(ds["emission_category_index"].values, dtype=np.int64).ravel()
        country_index = np.asarray(ds["country_index"].values, dtype=np.int64).ravel()
        lon_idx = np.asarray(ds["longitude_index"].values, dtype=np.int64).ravel()
        lat_idx = np.asarray(ds["latitude_index"].values, dtype=np.int64).ravel()
        nlon = int(ds.sizes["longitude"])
        nlat = int(ds.sizes["latitude"])
        lon_bounds = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
        lat_bounds = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
        if lon_idx.size and (lon_idx.max() >= nlon or lat_idx.max() >= nlat):
            log.debug("CAMS indices look 1-based; shifting to 0-based")
            lon_idx = lon_idx - 1
            lat_idx = lat_idx - 1
        np.clip(lon_idx, 0, nlon - 1, out=lon_idx)
        np.clip(lat_idx, 0, nlat - 1, out=lat_idx)
        pollutant_matrix = np.column_stack([
            np.nan_to_num(
                np.asarray(ds[cams_pollutant_var(lab)].values, dtype=np.float64).ravel(),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            for lab in pollutant_labels
        ])

    sector_base = (
        np.isfinite(lon_src)
        & np.isfinite(lat_src)
        & np.isin(emis_cat, ec_filter)
        & np.isin(src_type, st_filter)
    )
    has_emis = pollutant_matrix.max(axis=1) > 0.0

    country_sel = np.flatnonzero(sector_base & (country_index == int(country_idx)) & has_emis)
    country_cells = (
        np.unique(lat_idx[country_sel] * nlon + lon_idx[country_sel])
        if country_sel.size
        else np.array([], dtype=np.int64)
    )

    maritime_cells = _cells_centroid_in_maritime_domain(
        lon_bounds, lat_bounds, nlon, nlat, maritime_wgs84,
    )
    any_emis_sel = np.flatnonzero(sector_base & has_emis)
    cells_with_emis = np.unique(lat_idx[any_emis_sel] * nlon + lon_idx[any_emis_sel])
    maritime_cells = np.intersect1d(maritime_cells, cells_with_emis, assume_unique=False)

    footprint_cells = np.unique(np.concatenate([country_cells, maritime_cells]))
    if footprint_cells.size == 0:
        log.warning(
            f"No G_Shipping CAMS cells (country={iso3}, maritime buffer "
            f"{float(maritime_buffer_m):g} m)."
        )
        return {}, {
            "lon_bounds": np.asarray(lon_bounds, dtype=np.float64),
            "lat_bounds": np.asarray(lat_bounds, dtype=np.float64),
            "n_longitude": nlon,
            "n_latitude": nlat,
        }

    n_mar_only = int(np.setdiff1d(maritime_cells, country_cells).size)
    log.info(
        f"G_Shipping CAMS cells: {footprint_cells.size} total "
        f"({country_cells.size} country, {n_mar_only} maritime-only)"
    )

    mass_sel = np.flatnonzero(sector_base)
    cell_ids = lat_idx[mass_sel] * nlon + lon_idx[mass_sel]
    in_footprint = np.isin(cell_ids, footprint_cells)
    mass_sel = mass_sel[in_footprint]
    cell_ids = cell_ids[in_footprint]
    unique_cells, inverse = np.unique(cell_ids, return_inverse=True)
    sums = np.empty((unique_cells.size, len(pollutant_labels)), dtype=np.float64)
    for j in range(len(pollutant_labels)):
        sums[:, j] = np.bincount(
            inverse, weights=pollutant_matrix[mass_sel, j], minlength=unique_cells.size,
        )
    missing = np.setdiff1d(footprint_cells, unique_cells)
    if missing.size:
        unique_cells = np.concatenate([unique_cells, missing])
        sums = np.vstack([sums, np.zeros((missing.size, len(pollutant_labels)), dtype=np.float64)])

    out: dict[int, dict[str, Any]] = {}
    for k, cell_id in enumerate(unique_cells.tolist()):
        lo = int(cell_id % nlon)
        la = int(cell_id // nlon)
        west, east = float(lon_bounds[lo, 0]), float(lon_bounds[lo, 1])
        south, north = float(lat_bounds[la, 0]), float(lat_bounds[la, 1])
        out[int(cell_id)] = {
            "pollutants_within_cell": {
                lab: float(sums[k, j]) for j, lab in enumerate(pollutant_labels)
            },
            "longitude": 0.5 * (west + east),
            "latitude": 0.5 * (south + north),
            "width_deg": east - west,
            "height_deg": north - south,
            "cell_bounds_wgs84": {
                "west": west, "south": south, "east": east, "north": north,
            },
            "longitude_index": lo,
            "latitude_index": la,
            "n_longitude": nlon,
            "n_latitude": nlat,
            "year": int(year),
            "country_iso3": iso3,
        }

    grid = {
        "lon_bounds": np.asarray(lon_bounds, dtype=np.float64),
        "lat_bounds": np.asarray(lat_bounds, dtype=np.float64),
        "n_longitude": nlon,
        "n_latitude": nlat,
    }
    return out, grid
