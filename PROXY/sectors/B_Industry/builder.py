"""
B_Industry sector entrypoint: **reference grid** + **merged CEIP/group config** + shared
GNFR group pipeline (see `README.md` in this package).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PROXY.core.alpha import default_ceip_profile_relpath
from PROXY.core.dataloaders import project_root, resolve_path
from PROXY.core.grid import reference_window_profile
from PROXY.sectors._shared.gnfr_groups import merge_ceip_group_sector_cfg

logger = logging.getLogger(__name__)



def _log_domain_bbox(ref: dict[str, Any]) -> None:
    bbox = ref.get("domain_bbox_wgs84")
    if not bbox or len(bbox) != 4:
        return
    w, s, e, n = (float(x) for x in bbox)
    logger.info("B_Industry: reference domain WGS84 bbox (W,S,E,N): (%.4f, %.4f, %.4f, %.4f)", w, s, e, n)


def _merge_industry_pipeline_cfg(
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
    *,
    country: str,
    output_path: Path,
) -> dict[str, Any]:
    """
    Build the dict consumed by :func:`PROXY.sectors.B_Industry.pipeline.run_industry_pipeline`.

    * Path keys come from `path_cfg` + `industry.yaml` (see ``industry_paths``,
      ``ceip`` overrides).
    * Pollutant list normally lives in the YAML ``pollutants:``; if missing, the
      merge uses :data:`PROXY.sectors._shared.gnfr_groups.DEFAULT_CEIP_GROUP_POLLUTANTS`.
    * Duplicates the OSM path under ``osm_industry_gpkg`` for backwards-compatible
      access (same file as ``osm_group_gpkg`` in the merged struct).
    """
    merged = merge_ceip_group_sector_cfg(
        root=root,
        path_cfg=path_cfg,
        sector_cfg=sector_cfg,
        sector_paths_key="industry_paths",
        default_groups_yaml=default_ceip_profile_relpath(
            root, "B_Industry", "groups_yaml"
        ),
        osm_key="industry",
        output_path=output_path,
        profile_sector_id="B_Industry",
    )
    merged["paths"]["osm_industry_gpkg"] = merged["paths"]["osm_group_gpkg"]
    return merged


def build(*, path_cfg: dict[str, Any], sector_cfg: dict[str, Any], country: str) -> dict[str, Any]:
    """
    GNFR B industry area weights: CEIP group α, OSM + CORINE proxies, population blend,
    **normalize within CAMS cells** per group then per pollutant band.

    1. **Reference grid** — `reference_window_profile` on CORINE + NUTS for
       ``--country`` (output CRS / extent for all rasters in the shared pipeline).
    2. **Config merge** — CEIP paths, `pollutants`, `proxy` weights, output names.
    3. **Run** — :func:`run_industry_pipeline` in ``pipeline.py`` (delegates to
       :func:`PROXY.sectors._shared.gnfr_groups.run_gnfr_group_pipeline`).

    Parameters
    ----------
    path_cfg
        Resolved `paths.yaml` (proxy rasters, CAMS NetCDF, OSM `industry` GPKG, …).
    sector_cfg
        `industry.yaml` with `output_path` injected by `PROXY.main`.
    country
        CLI NUTS country / region (same convention as other sectors) for the window.

    Returns
    -------
    ``{"output_path": str}`` to the multiband GeoTIFF.
    """
    root = project_root()
    ctry = country.strip().upper()
    logger.info("B_Industry build: root=%s, cli country=%r", root, ctry)
    corine_path = resolve_path(root, path_cfg["proxy_common"]["corine_tif"])
    nuts_gpkg = resolve_path(root, path_cfg["proxy_common"]["nuts_gpkg"])
    pad_m = float(sector_cfg.get("pad_m", 5000.0))
    logger.info("B_Industry: CORINE (reference)=%s", corine_path)
    logger.info("B_Industry: NUTS gpkg=%s pad_m=%s", nuts_gpkg, pad_m)

    ref = reference_window_profile(
        corine_path=corine_path,
        nuts_gpkg=nuts_gpkg,
        nuts_country=ctry,
        pad_m=pad_m,
    )
    # Shared pipeline reads CORINE from ref when present; builder ensures it is set
    # after the same discovery path the grid helper used.
    ref["corine_path"] = str(corine_path.resolve())
    _log_domain_bbox(ref)

    output_path = Path(sector_cfg["output_path"]).resolve()
    industry_cfg = _merge_industry_pipeline_cfg(
        root,
        path_cfg,
        sector_cfg,
        country=country,
        output_path=output_path,
    )
    paths = industry_cfg.get("paths") or {}
    fb_iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()
    run_opts = sector_cfg.get("run") or {}
    show_p = bool(run_opts.get("show_progress", True))
    proxy_log = bool((industry_cfg.get("proxy") or {}).get("log_input_stats", True))
    logger.info("B_Industry: merged config — output TIF name=%s", output_path.name)
    logger.info("B_Industry: CAMS NetCDF = %s", resolve_path(root, paths.get("cams_nc")))
    logger.info("B_Industry: population = %s", resolve_path(root, paths.get("population_tif")))
    logger.info("B_Industry: OSM industry GPKG = %s", resolve_path(root, paths.get("osm_group_gpkg")))
    logger.info("B_Industry: CEIP workbook = %s", resolve_path(root, paths.get("ceip_workbook")))
    logger.info("B_Industry: CEIP groups YAML = %s", resolve_path(root, paths.get("ceip_groups_yaml")))
    logger.info("B_Industry: pollutants (band order) = %s", industry_cfg.get("pollutants"))
    logger.info(
        "B_Industry: country_iso3_fallback (CEIP / NUTS) = %s; run.show_progress = %s; "
        "proxy.log_input_stats = %s",
        fb_iso3,
        show_p,
        proxy_log,
    )

    from PROXY.sectors.B_Industry.pipeline import run_industry_pipeline

    logger.info("B_Industry: starting shared run_gnfr_group_pipeline (industry)…")
    out_tif = run_industry_pipeline(
        root,
        industry_cfg,
        ref,
        country_iso3_fallback=fb_iso3,
        show_progress=show_p,
    )
    out_resolved = Path(out_tif).resolve()
    if out_resolved != output_path:
        raise RuntimeError(
            f"Industry build wrote {out_resolved} but sector output_path is {output_path}"
        )
    logger.info("B_Industry build complete: %s", output_path)
    return {"output_path": str(output_path)}
