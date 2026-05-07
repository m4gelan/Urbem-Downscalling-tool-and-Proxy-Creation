"""GNFR G shipping area weights from EMODnet + CORINE ports + OSM (CAMS-cell normalized)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import rasterio
import yaml

from PROXY.sectors.G_Shipping.proxy_shipping import run_shipping_areasource
from PROXY.sectors.J_Waste.cams_waste_grid import build_cam_cell_id
from PROXY.sectors.J_Waste.normalization_waste import validate_weight_sums

from ..grid import resolve_path


def build_shipping_sourcearea(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
    *,
    sector_entry: dict[str, Any],
    run_validate: bool = False,
) -> Path:
    """
    Build ``Shipping_areasource.tif`` on the same CORINE+NUTS reference grid as other sectors.

    YAML defaults: ``Shipping/config/shipping_area.yaml`` (override via ``sector_entry["shipping_config"]``).
    """
    rel = Path(
        str(
            sector_entry.get("shipping_config")
            or "Shipping/config/shipping_area.yaml"
        )
    )
    cfg_path = rel if rel.is_absolute() else (root / rel)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Shipping config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        ship_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    paths = dict(ship_cfg.get("paths") or {})
    main_paths = cfg.get("paths") or {}
    for key in ("cams_nc", "corine", "nuts_gpkg"):
        if key in main_paths:
            paths[key] = main_paths[key]

    proxy_blk = ship_cfg.get("proxy") or {}
    osm_subdivide = int(proxy_blk.get("osm_subdivide", 4))
    write_diagnostics = bool(proxy_blk.get("write_diagnostics", False))

    cams_nc = resolve_path(root, Path(paths["cams_nc"]))
    emodnet = resolve_path(root, Path(paths["emodnet"]))
    osm_gpkg = resolve_path(root, Path(paths["osm_gpkg"]))

    out_dir = Path(cfg["output_dir"])
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_name = str(
        sector_entry.get("filename")
        or (ship_cfg.get("output") or {}).get("filename")
        or "Shipping_areasource.tif"
    )

    res = run_shipping_areasource(
        cams_nc=cams_nc,
        corine_path=None,
        osm_gpkg=osm_gpkg,
        emodnet_path=emodnet,
        output_folder=out_dir,
        ref=ref,
        fine_grid_tif=None,
        osm_subdivide=osm_subdivide,
        output_filename=out_name,
        write_diagnostics=write_diagnostics,
    )
    out_tif = Path(res["output_tif"])

    if run_validate:
        cam = build_cam_cell_id(cams_nc, ref).astype(np.int64)
        with rasterio.open(out_tif) as src:
            arr = src.read(1)
        errs = validate_weight_sums(arr, cam, None, tol=1e-3)
        if not errs:
            print(
                "validate: shipping_area OK (CAMS-cell sums ~ 1).",
                file=sys.stderr,
            )
        else:
            print(
                f"validate: shipping_area — {len(errs)} issue(s), e.g. {errs[:3]}",
                file=sys.stderr,
            )

    return out_tif
