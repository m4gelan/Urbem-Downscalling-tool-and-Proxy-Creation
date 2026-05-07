"""Unified ``build`` entrypoint for GNFR H (aviation).

Area-source products come from :mod:`PROXY.sectors.H_Aviation.aviation_area`.
Point-source matching remains on ``python -m PROXY.main match-points --sector H_Aviation``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PROXY.core.dataloaders import project_root, resolve_path
from PROXY.core.dataloaders.discovery import discover_cams_emissions
from PROXY.core.raster.country_clip import cams_iso3_from_cli_country
from PROXY.sectors.H_Aviation.aviation_area import run_aviation_area_proxy_pipeline

logger = logging.getLogger(__name__)


def build(*, path_cfg: dict[str, Any], sector_cfg: dict[str, Any], country: str) -> dict[str, Any]:
    """Run aviation **area-source** downscaling when enabled; honour ``build_part`` like other mixed sectors."""
    root = project_root()
    out_dir = resolve_path(root, Path(sector_cfg["output_dir"])).resolve()
    bp = str(sector_cfg.get("build_part", "both")).strip().lower()
    if bp not in ("both", "area", "point"):
        bp = "both"

    pm = sector_cfg.get("point_matching") if isinstance(sector_cfg.get("point_matching"), dict) else {}
    point_link_name = str(
        pm.get("link_geotiff_filename") or sector_cfg.get("output_filename_point") or "aviation_pointsource.tif"
    )

    if bp == "point":
        logger.info(
            "H_Aviation build --part=point: skipping area pipeline. "
            "Generate point matches / link GeoTIFF with: python -m PROXY.main match-points --sector H_Aviation ..."
        )
        return {"output_path": str(out_dir / point_link_name), "build_part": bp}

    path_resolved = dict(path_cfg)
    em = dict(path_resolved.get("emissions") or {})
    cams_raw = em.get("cams_2019_nc") or (path_cfg.get("emissions") or {}).get("cams_2019_nc")
    if not cams_raw:
        raise ValueError("paths.yaml emissions.cams_2019_nc missing")
    cams_nc = discover_cams_emissions(root, resolve_path(root, Path(str(cams_raw))))
    em["cams_2019_nc"] = str(cams_nc)
    path_resolved["emissions"] = em

    iso3 = str(pm.get("cams_country_iso3") or "").strip().upper()
    if not iso3:
        guess = cams_iso3_from_cli_country(country)
        if guess:
            iso3 = guess
    year = int(pm.get("reference_year", 2019))

    log_path = out_dir / f"aviation_area_run_{iso3 or 'CAMS'}_{year}.log"

    result = run_aviation_area_proxy_pipeline(
        repo_root=root,
        path_cfg=path_resolved,
        sector_cfg=sector_cfg,
        nuts_country=str(country),
        year=year,
        cams_iso3=iso3 if iso3 else None,
        log_file=log_path,
    )

    status = str(result.get("status", ""))
    area_name = str(sector_cfg.get("output_filename", "aviation_areasource.tif"))

    if status == "ok":
        ret = dict(result)
        ret["output_path"] = ret.get("proxy_tif") or str(out_dir / area_name)
        ret["build_part"] = bp
        return ret
    if status == "skipped_all":
        ret = dict(result)
        ret["output_path"] = ret.get("summary_csv") or str(out_dir / area_name)
        ret["build_part"] = bp
        logger.info(
            "H_Aviation area build: no pollutants with CAMS area mass — summary only. "
            "Point matching unchanged (see match-points)."
        )
        return ret
    if status == "disabled":
        logger.info("H_Aviation area_source.enabled is false; no area raster written.")
        return {"output_path": str(out_dir / area_name), "build_part": bp, **result}

    ret = dict(result)
    ret.setdefault("output_path", str(out_dir / area_name))
    ret["build_part"] = bp
    return ret
