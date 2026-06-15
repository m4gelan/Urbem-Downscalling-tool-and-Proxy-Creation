from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


OUTPUT_RESOLUTIONS = (100, 1000)


@dataclass
class NativeGridMeta:
    res_x: float
    res_y: float
    transform: rasterio.Affine
    crs: str


@dataclass
class FineGrid:
    transform: rasterio.Affine
    crs: str
    height: int
    width: int
    domain_mask: np.ndarray  # bool (H, W)

    @property
    def size(self) -> int:
        return int(self.height * self.width)


def native_grid_metadata(path: Path) -> NativeGridMeta:
    with rasterio.open(path) as src:
        return NativeGridMeta(
            res_x=abs(float(src.transform.a)),
            res_y=abs(float(src.transform.e)),
            transform=src.transform,
            crs=str(src.crs),
        )


def validate_output_resolution(resolution_m: int, native: NativeGridMeta) -> None:
    if resolution_m not in OUTPUT_RESOLUTIONS:
        raise ValueError(f"output.grid_resolution_m must be one of {OUTPUT_RESOLUTIONS}, got {resolution_m}")
    factor = resolution_m / native.res_x
    if abs(factor - round(factor)) > 1e-6 or factor < 1.0:
        raise ValueError(
            f"output.grid_resolution_m ({resolution_m}) must be an integer multiple of "
            f"native weight resolution ({native.res_x})"
        )


def build_output_grid(domain: dict, resolution_m: int, native: NativeGridMeta) -> FineGrid:
    validate_output_resolution(resolution_m, native)
    xmin = float(domain["xmin"])
    ymin = float(domain["ymin"])
    xmax = float(domain["xmax"])
    ymax = float(domain["ymax"])
    crs = str(domain["crs"])
    res = float(resolution_m)

    if resolution_m == int(native.res_x):
        w = max(1, int(np.ceil((xmax - xmin) / native.res_x)))
        h = max(1, int(np.ceil((ymax - ymin) / native.res_y)))
        transform = from_bounds(xmin, ymin, xmax, ymax, w, h)
        mask = np.ones((h, w), dtype=bool)
        return FineGrid(transform=transform, crs=crs, height=h, width=w, domain_mask=mask)

    anchor_x = float(native.transform.c)
    anchor_y = float(native.transform.f)
    col0 = int(np.floor((xmin - anchor_x) / res))
    col1 = int(np.ceil((xmax - anchor_x) / res))
    row0 = int(np.floor((anchor_y - ymax) / res))
    row1 = int(np.ceil((anchor_y - ymin) / res))
    width = max(1, col1 - col0)
    height = max(1, row1 - row0)
    xmin_out = anchor_x + col0 * res
    xmax_out = anchor_x + col1 * res
    ymax_out = anchor_y - row0 * res
    ymin_out = anchor_y - row1 * res
    transform = from_bounds(xmin_out, ymin_out, xmax_out, ymax_out, width, height)

    px = transform.c + (np.arange(width, dtype=np.float64) + 0.5) * transform.a
    py = transform.f + (np.arange(height, dtype=np.float64) + 0.5) * transform.e
    yy, xx = np.meshgrid(py, px, indexing="ij")
    mask = (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
    return FineGrid(transform=transform, crs=crs, height=height, width=width, domain_mask=mask)


def fine_grid_from_reference(path: Path, domain: dict) -> FineGrid:
    """Build target grid from reference GeoTIFF resolution and domain bbox."""
    native = native_grid_metadata(path)
    return build_output_grid(domain, int(native.res_x), native)


def resolve_path(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (root / p).resolve()


def find_reference_tif(config: dict, sector_order: list[str]) -> Path | None:
    from UrbEm_Visualizer.paths import project_root

    base = project_root()
    sectors = config.get("sectors") or {}
    for sid in sector_order:
        sec = sectors.get(sid) or {}
        aw = sec.get("area_weights") or {}
        cats = aw.get("categories")
        if cats:
            for p in cats.values():
                if p:
                    return resolve_path(p, base)
        if aw.get("path"):
            return resolve_path(aw["path"], base)
    for sid in sector_order:
        sec = sectors.get(sid) or {}
        ps = sec.get("point_source") or {}
        if ps.get("path"):
            return resolve_path(ps["path"], base)
    return None
