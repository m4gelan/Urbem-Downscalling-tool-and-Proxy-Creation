"""
prepare_cams_emissions - Read CAMS-REG-ANT v8.1 NetCDF, filter by sectors,
accumulate emissions into lon/lat grid, return raster stacks per sector.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from urbem_interface.logging_config import get_logger

logger = get_logger(__name__)

GNFR_CODES = [
    "A", "B", "C", "D", "E",
    "F1", "F2", "F3", "F4",
    "G", "H", "I", "J", "K", "L",
]

GNFR_FULL_NAMES = [
    "A_PublicPower", "B_Industry", "C_OtherStationaryComb", "D_Fugitives",
    "E_Solvents",
    "F1_RoadTransport_Exhaust_Gasoline", "F2_RoadTransport_Exhaust_Diesel",
    "F3_RoadTransport_Exhaust_LPG_gas", "F4_RoadTransport_NonExhaust",
    "G_Shipping", "H_Aviation", "I_OffRoad", "J_Waste",
    "K_AgriLivestock", "L_AgriOther",
]

POLLUTANTS = ["ch4", "co", "nh3", "nmvoc", "nox", "pm10", "pm2_5", "sox"]

POLLUTANT_NAMES = {
    "ch4": "CH4", "co": "CO", "nh3": "NH3", "nmvoc": "NMVOC",
    "nox": "NOx", "pm10": "PM10", "pm2_5": "PM2.5", "sox": "SO2",
}


@dataclass
class CamsRasterResult:
    """Raster stacks per sector with grid metadata."""
    rasters: dict[str, dict[str, np.ndarray]]
    sectors: list[int]
    sector_codes: list[str]
    sector_names: list[str]
    pollutants: list[str]
    source_type: str
    grid_lons: np.ndarray
    grid_lats: np.ndarray


def _resolve_sectors(sectors: list[str] | None, emis_cat_names: list) -> list[int]:
    if sectors is None:
        return list(range(1, len(GNFR_CODES) + 1))
    indices = []
    for s in sectors:
        code = s.split("_")[0] if "_" in s else s
        idx = next((i + 1 for i, c in enumerate(GNFR_CODES) if c == code), None)
        if idx is not None:
            indices.append(idx)
    return indices


def prepare_cams_emissions(
    nc_file_path: str | Path,
    source_type: str = "area",
    pollutants: list[str] | None = None,
    sectors: list[str] | None = None,
) -> CamsRasterResult:
    """Read CAMS v8.1 NetCDF, filter, bin to grid, accumulate per sector."""
    if pollutants is None:
        pollutants = list(POLLUTANTS)
    else:
        pollutants = [p for p in pollutants if p in POLLUTANT_NAMES]
    if not pollutants:
        logger.error("No valid pollutants specified")

    logger.info(f"Preparing CAMS emissions for {nc_file_path}")
    ds = xr.open_dataset(str(nc_file_path))

    logger.info("    Reading source indices...")
    emis_cat = np.asarray(ds["emission_category_index"].values).ravel().astype(int)
    src_type  = np.asarray(ds["source_type_index"].values).ravel().astype(int)
    lon_src   = np.asarray(ds["longitude_source"].values).ravel()
    lat_src   = np.asarray(ds["latitude_source"].values).ravel()

    cams_lon = np.asarray(ds["longitude"].values).ravel()
    cams_lat = np.asarray(ds["latitude"].values).ravel()
    nlon, nlat = len(cams_lon), len(cams_lat)

    sector_indices = _resolve_sectors(sectors, list(ds.get("emis_cat_name", [])))
    if not sector_indices:
        raise ValueError("No valid sectors specified")

    if source_type == "area":
        source_mask = src_type == 1
    elif source_type == "point":
        source_mask = src_type == 2
    else:
        source_mask = np.ones(len(src_type), dtype=bool)

    sector_mask = np.isin(emis_cat, sector_indices)
    combined    = source_mask & sector_mask

    filtered_cat = emis_cat[combined]
    logger.debug(f"Building for source type: {source_type}")
    logger.debug(f"Sector indices are : {sector_indices} ")

    if "longitude_index" in ds and "latitude_index" in ds:
        lon_idx_raw = np.asarray(ds["longitude_index"].values).ravel().astype(int)[combined]
        lat_idx_raw = np.asarray(ds["latitude_index"].values).ravel().astype(int)[combined]
        if lon_idx_raw.max() >= nlon or lat_idx_raw.max() >= nlat:
            lon_idx_raw = np.maximum(0, lon_idx_raw - 1)
            lat_idx_raw = np.maximum(0, lat_idx_raw - 1)
        lon_idx = np.clip(lon_idx_raw, 0, nlon - 1)
        lat_idx = np.clip(lat_idx_raw, 0, nlat - 1)
    else:
        filtered_lons = lon_src[combined]
        filtered_lats = lat_src[combined]
        lon_step = float(cams_lon[1] - cams_lon[0]) if nlon > 1 else 1.0
        lat_step = float(cams_lat[1] - cams_lat[0]) if nlat > 1 else 1.0
        lon_idx = np.clip(
            np.round((filtered_lons - cams_lon[0]) / lon_step).astype(int), 0, nlon - 1
        )
        lat_idx = np.clip(
            np.round((filtered_lats - cams_lat[0]) / lat_step).astype(int), 0, nlat - 1
        )

    flat_grid_idx = lat_idx * nlon + lon_idx

    sector_indices_arr = np.asarray(sector_indices)
    cat_to_label = np.full(sector_indices_arr.max() + 1, -1, dtype=np.int32)
    for label, idx in enumerate(sector_indices_arr):
        cat_to_label[idx] = label
    sector_label = cat_to_label[filtered_cat]

    n_sectors  = len(sector_indices)
    grid_cells = nlat * nlon

    combined_idx = sector_label * grid_cells + flat_grid_idx
    total_bins   = n_sectors * grid_cells

    emission_block: dict[str, np.ndarray] = {}

    for v81_key in pollutants:
        if v81_key not in ds:
            logger.debug(f"Pollutant {v81_key} not in NetCDF, skipping")
            continue
        arr = np.asarray(ds[v81_key].values).ravel()
        if arr.size != combined.size:
            logger.debug(f"Pollutant {v81_key} size mismatch, skipping")
            continue

        vals = arr[combined]

        binned = np.bincount(
            combined_idx,
            weights=vals,
            minlength=total_bins,
        ).reshape(n_sectors, nlat, nlon)

        emission_block[v81_key] = binned
        logger.debug(f"Binning {v81_key} done")

    ds.close()
    logger.info("Finished preparing CAMS emissions")

    lat_ascending = (cams_lat[0] < cams_lat[-1]) if nlat > 1 else True
    if lat_ascending:
        cams_lat = cams_lat[::-1]
        for v81_key in emission_block:
            emission_block[v81_key] = emission_block[v81_key][:, ::-1, :]

    sector_rasters: dict[str, dict[str, np.ndarray]] = {}
    for label, i in enumerate(sector_indices):
        name = GNFR_FULL_NAMES[i - 1]
        sector_rasters[name] = {
            POLLUTANT_NAMES[p]: emission_block[p][label].copy()
            for p in pollutants
            if p in emission_block
        }
    return CamsRasterResult(
        rasters=sector_rasters,
        sectors=sector_indices,
        sector_codes=[GNFR_CODES[i - 1] for i in sector_indices],
        sector_names=[GNFR_FULL_NAMES[i - 1] for i in sector_indices],
        pollutants=pollutants,
        source_type=source_type,
        grid_lons=cams_lon,
        grid_lats=cams_lat,
    )
