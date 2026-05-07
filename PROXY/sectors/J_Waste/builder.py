from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PROXY.core.dataloaders import project_root, resolve_path

from PROXY.sectors.J_Waste.pipeline import merge_waste_pipeline_cfg, run_waste_pipeline

logger = logging.getLogger(__name__)


def build(*, path_cfg: dict[str, Any], sector_cfg: dict[str, Any], country: str) -> dict[str, Any]:
    """
    GNFR J waste area + point within-cell weights (CEIP family shares, spatial proxies, CAMS cells).

    Configuration: ``sector_cfg["waste"]`` overlays
    ``PROXY/config/ceip/profiles/waste_pipeline.yaml``. Alpha uses the
    reported-emissions workbook (``paths.proxy_common.alpha_workbook``) by default, with
    mean mass over years; optional legacy ``waste.ceip_xlsx`` if the workbook is absent.
    """
    root = project_root()
    out_dir = resolve_path(root, Path(sector_cfg["output_dir"])).resolve()
    logger.info("J_Waste build: country=%s output_dir=%s", country, out_dir)

    cfg = merge_waste_pipeline_cfg(
        root,
        path_cfg,
        sector_cfg,
        country=country,
        output_dir=out_dir,
    )
    bp = str(sector_cfg.get("build_part", "both")).strip().lower()
    if bp not in ("both", "area", "point"):
        bp = "both"
    cfg["waste_build_part"] = bp
    paths = cfg.get("paths") or {}
    logger.info(
        "J_Waste build: merged cfg cams=%s corine=%s osm=%s ceip=%s",
        paths.get("cams_nc"),
        paths.get("corine"),
        paths.get("osm_waste_gpkg"),
        paths.get("ceip_workbook"),
    )
    out = run_waste_pipeline(root, cfg)
    out_area = Path(out["output_tif_area"]).resolve()
    out_point = Path(out["output_tif_point"]).resolve()
    wrote_a = bool(out.get("wrote_area", True))
    wrote_p = bool(out.get("wrote_point", True))
    if wrote_a and not out_area.is_file():
        raise RuntimeError(f"J_Waste build did not write area output: {out_area}")
    if wrote_p and not out_point.is_file():
        raise RuntimeError(f"J_Waste build did not write point output: {out_point}")
    logger.info("J_Waste build complete: area=%s point=%s", out_area, out_point)
    ret: dict[str, Any] = {"output_dir": out["output_dir"], "build_part": bp}
    if wrote_a:
        ret["output_path"] = str(out_area)
    if wrote_p:
        ret["output_path_point"] = str(out_point)
    if not ret.get("output_path") and ret.get("output_path_point"):
        ret["output_path"] = str(out_point)
    elif not ret.get("output_path"):
        ret["output_path"] = str(out_area)
    return ret
