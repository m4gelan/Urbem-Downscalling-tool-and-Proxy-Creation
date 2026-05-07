from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


@dataclass(frozen=True)
class TemplateGrid:
    """Output grid aligned with the CORINE 100 m raster used as the build template."""

    crs: CRS
    transform: Affine
    width: int
    height: int

    @property
    def projected_bbox(self) -> dict[str, float]:
        """Axis-aligned bounds in the grid CRS (typically EPSG:3035)."""
        t = self.transform
        xmin = t.c
        ymax = t.f
        xmax = xmin + self.width * t.a
        ymin = ymax + self.height * t.e
        return {"xmin": float(xmin), "ymin": float(ymin), "xmax": float(xmax), "ymax": float(ymax)}


def load_grid_from_corine_raster(corine_raster: Path) -> TemplateGrid:
    """Define the output grid from the CORINE GeoTIFF (native CRS, usually EPSG:3035, 100 m)."""
    corine_raster = Path(corine_raster)
    if not corine_raster.exists():
        raise FileNotFoundError(f"CORINE raster not found: {corine_raster}")
    with rasterio.open(corine_raster) as ds:
        if not ds.crs:
            raise ValueError(f"CORINE raster has no CRS: {corine_raster}")
        return TemplateGrid(
            crs=ds.crs,
            transform=ds.transform,
            width=int(ds.width),
            height=int(ds.height),
        )
