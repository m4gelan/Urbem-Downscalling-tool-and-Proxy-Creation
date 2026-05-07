from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr


def open_cams_dataset(path: Path) -> xr.Dataset:
    return xr.open_dataset(path, engine="netcdf4")


def dataset_dims(path: Path) -> dict[str, int]:
    with xr.open_dataset(path, engine="netcdf4") as ds:
        return {str(k): int(v) for k, v in ds.dims.items()}


def dataset_variables(path: Path) -> list[str]:
    with xr.open_dataset(path, engine="netcdf4") as ds:
        return sorted([str(k) for k in ds.data_vars.keys()])


def dataset_summary(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "dims": dataset_dims(path),
        "variables": dataset_variables(path),
    }

