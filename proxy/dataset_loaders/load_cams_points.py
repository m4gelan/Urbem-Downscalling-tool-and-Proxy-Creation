from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from proxy.core import log
from proxy.core.alias import cams_country_index_from_iso3, cams_pollutant_var


def load_cams_points(
    cams_nc: Path,
    *,
    year: int,
    country_iso3: str,
    emission_category_indices: list[int],
    source_type_indices: list[int],
    pollutants: list[str],
) -> dict[int, dict[str, Any]]:
    """
    Point sources: filter by emission/source type and CAMS ``country_index`` (via ISO3 on ``country_id``).

    Returns ``{point_id: {"latitude", "longitude", "pollutants": {label: value, ...}, "year"}}``.
    """
    iso3 = str(country_iso3).strip().upper()
    ec_filter = np.asarray(emission_category_indices, dtype=np.int64)
    st_filter = np.asarray(source_type_indices, dtype=np.int64)

    with xr.open_dataset(cams_nc, engine="netcdf4") as cams_dataset:
        country_idx = cams_country_index_from_iso3(cams_dataset, iso3)
        lon = np.asarray(cams_dataset["longitude_source"].values).ravel().astype(np.float64)
        lat = np.asarray(cams_dataset["latitude_source"].values).ravel().astype(np.float64)

        source_type_indices = np.asarray(cams_dataset["source_type_index"].values).ravel().astype(np.int64)
        emission_category_indices = np.asarray(cams_dataset["emission_category_index"].values).ravel().astype(np.int64)
        country_index = np.asarray(cams_dataset["country_index"].values).ravel().astype(np.int64)

        pol_m: dict[str, np.ndarray] = {}
        for label in pollutants:
            vname = cams_pollutant_var(label)
            arr = np.asarray(cams_dataset[vname].values).ravel().astype(np.float64)
            pol_m[label] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        mask = np.isfinite(lon) & np.isfinite(lat)
        mask &= np.isin(emission_category_indices, ec_filter)
        mask &= np.isin(source_type_indices, st_filter)
        mask &= country_index == int(country_idx)
        if pol_m:
            mask &= np.max(np.stack(list(pol_m.values()), axis=1), axis=1) > 0.0

        if not np.any(mask):
            log.error("No cams points source for this sector for this country, consider desabling the Point source matching")
            SystemExit(1)
       
        # Create the output dictionary
        out: dict[int, dict[str, Any]] = {}
        for i in np.flatnonzero(mask):
            pid = int(i)
            out[pid] = {
                "latitude": float(lat[i]),
                "longitude": float(lon[i]),
                "pollutants": {lab: float(pol_m[lab][i]) for lab in pol_m},
                "year": int(year),
            }

    nout = len(out)
    log.info(f"Total CAMS point sources (after filters): {nout}")
    for j, (pid, row) in enumerate(out.items()):
        if j >= 1:
            break
        vals = [f"{k}={v/1_000_000:.3f} Gg.yr-1" for k, v in row["pollutants"].items() if v > 0]
        log.debug(
            f" Example point: ID: {pid} "
            f"lat={row['latitude']:.4f} lon={row['longitude']:.4f} "
            f"pollutants={{ {', '.join(vals)} }}"
        )
    log.debug(f"CAMS load complete: {nout} points from {cams_nc.name}")
   
    return out
