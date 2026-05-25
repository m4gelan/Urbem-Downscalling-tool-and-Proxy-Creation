from __future__ import annotations

import numpy as np
import xarray as xr


def _add_da(stack: np.ndarray, da: xr.DataArray, pollutants: list[str]) -> None:
    for i, pol in enumerate(pollutants):
        if pol in da.coords["pollutant"].values:
            stack[i] += da.sel(pollutant=pol).values.astype(np.float32)


def merge_grids(
    area: xr.DataArray | None,
    point: xr.DataArray | None,
    pollutants: list[str],
    shape: tuple[int, int],
) -> xr.DataArray:
    h, w = shape
    n = len(pollutants)
    stack = np.zeros((n, h, w), dtype=np.float32)
    for src in (area, point):
        if src is not None:
            _add_da(stack, src, pollutants)
    return xr.DataArray(
        stack,
        dims=("pollutant", "y", "x"),
        coords={"pollutant": pollutants, "y": np.arange(h), "x": np.arange(w)},
    )
