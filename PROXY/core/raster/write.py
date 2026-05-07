"""Multiband GeoTIFF writer with consistent tags and band descriptions.

Thin wrapper around :func:`PROXY.core.io.write_geotiff`. The purpose is to standardize
the metadata (``Software``, ``Sector``, ``Pollutants``) so every sector's output has a
predictable header.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from PROXY.core.io import write_geotiff


def write_multiband_geotiff(
    path: Path,
    bands: Mapping[str, np.ndarray],
    *,
    ref: Mapping[str, Any],
    dtype: str = "float32",
    nodata: float | int | None = None,
    predictor: int | None = None,
    tiled: bool = False,
    bigtiff: str | None = None,
    tags: Mapping[str, str] | None = None,
    sector: str | None = None,
    pollutants: list[str] | None = None,
    sources: Mapping[str, str] | None = None,
) -> Path:
    """Write a multiband GeoTIFF with band descriptions set to pollutant keys.

    Parameters
    ----------
    path
        Output GeoTIFF path. Parent directories are created.
    bands
        Mapping ``band_name -> 2D array``. All arrays must share shape ``(H, W)``. The
        iteration order of ``bands`` defines the output band order.
    ref
        Reference grid dict (``height``, ``width``, ``transform``, ``crs``).
    dtype
        Numpy dtype string for the output.
    nodata
        Optional nodata sentinel written to the file profile.
    predictor, tiled, bigtiff
        Passed through to ``core.io.write_geotiff``.
    tags
        Optional extra file-level tags. ``Software``, ``Sector`` and ``Pollutants`` are
        always set by this helper.
    sector, pollutants, sources
        Convenience fields; merged into ``tags``.
    """
    if not bands:
        raise ValueError("write_multiband_geotiff requires at least one band.")

    band_names = list(bands.keys())
    arr_list = [np.asarray(bands[k]) for k in band_names]
    shapes = {a.shape for a in arr_list}
    if len(shapes) != 1:
        raise ValueError(f"All bands must share the same shape; got {shapes!r}.")
    stack = np.stack(arr_list, axis=0).astype(dtype, copy=False)

    merged_tags: dict[str, str] = {"Software": "PROXY"}
    if sector is not None:
        merged_tags["Sector"] = str(sector)
    merged_tags["Pollutants"] = ",".join(band_names) if pollutants is None else ",".join(pollutants)
    if sources:
        merged_tags.update({f"Source-{k}": str(v) for k, v in sources.items()})
    if tags:
        merged_tags.update({str(k): str(v) for k, v in tags.items()})

    return write_geotiff(
        path=path,
        array=stack,
        crs=str(ref["crs"]),
        transform=ref["transform"],
        nodata=nodata,
        tags=merged_tags,
        band_descriptions=band_names,
        tiled=tiled,
        predictor=predictor,
        bigtiff=bigtiff,
    )
