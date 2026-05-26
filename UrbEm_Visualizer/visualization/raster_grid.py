from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from affine import Affine
from rasterio.transform import array_bounds


@dataclass
class AreaRaster:
    data: np.ndarray
    transform: Affine
    crs: str
    nodata: float = 0.0

    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.data.shape
        left, bottom, right, top = array_bounds(h, w, self.transform)
        return left, bottom, right, top


@dataclass
class GridTemplate:
    xs: np.ndarray
    ys: np.ndarray
    transform: Affine
    crs: str
    x_index: dict[float, int]
    y_index: dict[float, int]

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ys), len(self.xs)


def build_template(domain: dict, frames: list[pd.DataFrame]) -> GridTemplate:
    xs_set: set[float] = set()
    ys_set: set[float] = set()
    for df in frames:
        if df.empty:
            continue
        xs_set.update(df["x"].astype(float).unique())
        ys_set.update(df["y"].astype(float).unique())
    if not xs_set or not ys_set:
        xmin, ymin, xmax, ymax = (
            float(domain["xmin"]),
            float(domain["ymin"]),
            float(domain["xmax"]),
            float(domain["ymax"]),
        )
        ncol, nrow = 64, 64
        dx = (xmax - xmin) / ncol
        dy = (ymax - ymin) / nrow
        xs = xmin + (np.arange(ncol) + 0.5) * dx
        ys = ymax - (np.arange(nrow) + 0.5) * dy
    else:
        xs = np.sort(np.asarray(list(xs_set), dtype=np.float64))
        ys = np.sort(np.asarray(list(ys_set), dtype=np.float64))[::-1]

    nrow, ncol = len(ys), len(xs)
    dx = float(np.median(np.diff(xs))) if ncol > 1 else (float(domain["xmax"]) - float(domain["xmin"])) / max(ncol, 1)
    dy = float(np.median(np.abs(np.diff(ys)))) if nrow > 1 else (float(domain["ymax"]) - float(domain["ymin"])) / max(nrow, 1)
    x0 = float(xs[0]) - dx * 0.5
    y0 = float(ys[0]) + dy * 0.5
    transform = Affine(dx, 0.0, x0, 0.0, -dy, y0)
    x_index = {float(v): i for i, v in enumerate(xs)}
    y_index = {float(v): i for i, v in enumerate(ys)}
    return GridTemplate(
        xs=xs,
        ys=ys,
        transform=transform,
        crs=str(domain["crs"]),
        x_index=x_index,
        y_index=y_index,
    )


def df_to_raster(df: pd.DataFrame, template: GridTemplate) -> AreaRaster:
    arr = np.zeros(template.shape, dtype=np.float32)
    if not df.empty:
        for rec in df.itertuples(index=False):
            xi = template.x_index.get(float(rec.x))
            yi = template.y_index.get(float(rec.y))
            if xi is not None and yi is not None:
                arr[yi, xi] += np.float32(rec.emission)
    return AreaRaster(data=arr, transform=template.transform, crs=template.crs)


def sum_rasters(rasters: list[AreaRaster], template: GridTemplate) -> AreaRaster:
    out = np.zeros(template.shape, dtype=np.float32)
    for r in rasters:
        if r.data.shape == out.shape:
            out += r.data
    return AreaRaster(data=out, transform=template.transform, crs=template.crs)
