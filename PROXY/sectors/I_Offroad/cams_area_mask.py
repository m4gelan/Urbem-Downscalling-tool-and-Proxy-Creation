"""Visualization-facing CAMS area-source mask for I_Offroad.

The production weight build uses ``core.cams.grid.build_cam_cell_id`` directly. This
module remains for ``PROXY.visualization.offroad_area_map``, where the user can display
the configured GNFR I area-source category union on the map.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from PROXY.core.cams.domain import country_index_1based, domain_mask_wgs84


def offroad_union_area_mask(
    ds: xr.Dataset,
    iso3: str,
    *,
    emission_category_indices: tuple[int, ...],
    source_type_index: int = 1,
    domain_bbox_wgs84: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """True where CAMS lists any of ``emission_category_indices`` as area sources for ``iso3``."""
    cidx = country_index_1based(ds, iso3)
    emis = np.asarray(ds["emission_category_index"].values).ravel().astype(np.int64)
    ci = np.asarray(ds["country_index"].values).ravel().astype(np.int64)
    st = np.asarray(ds["source_type_index"].values).ravel().astype(np.int64)
    lon = np.asarray(ds["longitude_source"].values).ravel()
    lat = np.asarray(ds["latitude_source"].values).ravel()
    cats = [int(x) for x in emission_category_indices]
    base = (
        np.isin(emis, cats)
        & (st == int(source_type_index))
        & domain_mask_wgs84(lon, lat, ci, cidx, domain_bbox_wgs84)
    )
    return base
