from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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


def _template_from_fine_grid(fine) -> GridTemplate:
    xs = fine.transform.c + (np.arange(fine.width, dtype=np.float64) + 0.5) * fine.transform.a
    ys = fine.transform.f + (np.arange(fine.height, dtype=np.float64) + 0.5) * fine.transform.e
    x_index = {float(v): i for i, v in enumerate(xs)}
    y_index = {float(v): i for i, v in enumerate(ys)}
    return GridTemplate(
        xs=xs,
        ys=ys,
        transform=fine.transform,
        crs=fine.crs,
        x_index=x_index,
        y_index=y_index,
    )


def grid_from_manifest(output_dir: Path) -> GridTemplate | None:
    path = output_dir / "manifest.yaml"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    grid = raw.get("grid") if isinstance(raw, dict) else None
    if not isinstance(grid, dict):
        return None
    tr = grid.get("transform")
    height = grid.get("height")
    width = grid.get("width")
    crs = grid.get("crs")
    if not (isinstance(tr, list) and len(tr) == 6 and height and width and crs):
        return None
    transform = Affine(*[float(v) for v in tr])
    h, w = int(height), int(width)
    xs = transform.c + (np.arange(w, dtype=np.float64) + 0.5) * transform.a
    ys = transform.f + (np.arange(h, dtype=np.float64) + 0.5) * transform.e
    return GridTemplate(
        xs=xs,
        ys=ys,
        transform=transform,
        crs=str(crs),
        x_index={float(v): i for i, v in enumerate(xs)},
        y_index={float(v): i for i, v in enumerate(ys)},
    )


def build_template(
    domain: dict,
    frames: list[pd.DataFrame],
    *,
    config: dict | None = None,
    output_dir: Path | None = None,
) -> GridTemplate:
    if output_dir is not None:
        from_manifest = grid_from_manifest(output_dir)
        if from_manifest is not None:
            return from_manifest

    if config is not None:
        output_cfg = config.get("output") or {}
        if "grid_resolution_m" in output_cfg:
            from UrbEm_Visualizer.downscaling.sector_meta import sector_order
            from UrbEm_Visualizer.downscaling.spatial import (
                build_output_grid,
                find_reference_tif,
                native_grid_metadata,
            )

            ref = find_reference_tif(config, sector_order(config))
            if ref is not None:
                resolution_m = int(output_cfg["grid_resolution_m"])
                fine = build_output_grid(domain, resolution_m, native_grid_metadata(ref))
                return _template_from_fine_grid(fine)

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
    if df.empty:
        return AreaRaster(data=arr, transform=template.transform, crs=template.crs)
    inv = ~template.transform
    h, w = template.shape
    for rec in df.itertuples(index=False):
        col_f, row_f = inv * (float(rec.x), float(rec.y))
        col = int(np.floor(col_f))
        row = int(np.floor(row_f))
        if 0 <= row < h and 0 <= col < w:
            arr[row, col] += np.float32(rec.emission)
    return AreaRaster(data=arr, transform=template.transform, crs=template.crs)


def sum_rasters(rasters: list[AreaRaster], template: GridTemplate) -> AreaRaster:
    out = np.zeros(template.shape, dtype=np.float32)
    for r in rasters:
        if r.data.shape == out.shape:
            out += r.data
    return AreaRaster(data=out, transform=template.transform, crs=template.crs)
