from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds


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


def fine_grid_from_reference(path: Path, domain: dict) -> FineGrid:
    """Build target grid from reference GeoTIFF resolution and domain bbox."""
    xmin = float(domain["xmin"])
    ymin = float(domain["ymin"])
    xmax = float(domain["xmax"])
    ymax = float(domain["ymax"])
    crs = str(domain["crs"])

    with rasterio.open(path) as src:
        res_x = abs(float(src.transform.a))
        res_y = abs(float(src.transform.e))

    w = max(1, int(np.ceil((xmax - xmin) / res_x)))
    h = max(1, int(np.ceil((ymax - ymin) / res_y)))
    transform = from_bounds(xmin, ymin, xmax, ymax, w, h)
    mask = np.ones((h, w), dtype=bool)
    return FineGrid(transform=transform, crs=crs, height=h, width=w, domain_mask=mask)


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
