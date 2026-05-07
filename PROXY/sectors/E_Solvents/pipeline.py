"""Merge GNFR E solvent config and expose the public pipeline runner."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from PROXY.core.ceip import default_ceip_profile_relpath, remap_legacy_ceip_relpath
from PROXY.core.dataloaders import resolve_path
from PROXY.core.dataloaders.discovery import discover_cams_emissions, discover_corine
from PROXY.core.grid import resolve_nuts_cntr_code


def _load_solvents_base(root: Path) -> dict[str, Any]:
    candidates = [
        root / "PROXY" / "config" / "ceip" / "profiles" / "solvents_pipeline.yaml",
        root / "PROXY" / "config" / "solvents" / "defaults.json",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        if p.suffix.lower() in (".yaml", ".yml"):
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(
        "E_Solvents base config not found: expected PROXY/config/ceip/profiles/solvents_pipeline.yaml "
        "or legacy PROXY/config/solvents/defaults.json"
    )


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def merge_solvents_pipeline_cfg(
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
    *,
    country: str,
    output_path: Path,
) -> dict[str, Any]:
    """
    Build cfg dict for :func:`run_solvents_pipeline`.

    Resolves PROXY ``paths.yaml`` and ``sector_cfg["solvents"]`` overlays.
    """
    cfg = deepcopy(_load_solvents_base(root))
    sp = sector_cfg.get("solvents") or {}
    ceip_sp = sp.get("ceip") if isinstance(sp.get("ceip"), dict) else {}
    pcommon = path_cfg.get("proxy_common") or {}
    paths_overlay = sp.get("paths") if isinstance(sp.get("paths"), dict) else {}

    corine_configured = resolve_path(root, Path(pcommon["corine_tif"]))
    corine = discover_corine(root, corine_configured)
    cams = discover_cams_emissions(
        root, resolve_path(root, Path(path_cfg["emissions"]["cams_2019_nc"]))
    )
    nuts_gpkg = resolve_path(root, pcommon["nuts_gpkg"])
    pop_tif = resolve_path(root, pcommon["population_tif"])
    osm_gpkg = resolve_path(root, path_cfg["osm"]["solvents"])

    iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()
    nuts_override = str(sector_cfg.get("nuts_cntr", "")).strip().upper()
    nuts_cntr = nuts_override if len(nuts_override) == 2 else resolve_nuts_cntr_code(country)

    wb_rel = (
        paths_overlay.get("ceip_workbook")
        or ceip_sp.get("workbook")
        or ceip_sp.get("ceip_workbook")
        or sp.get("ceip_workbook")
        or pcommon.get("alpha_workbook")
        or "INPUT/Proxy/Alpha/Reported_Emissions_EU27_2018_2023.xlsx"
    )
    ceip_workbook = resolve_path(root, Path(str(wb_rel)))

    submap_rel = (
        paths_overlay.get("ceip_subsector_map_yaml")
        or ceip_sp.get("ceip_subsector_map_yaml")
        or sp.get("ceip_subsector_map_yaml")
        or default_ceip_profile_relpath(root, "E_Solvents", "subsectors_yaml")
    )
    submap_path = resolve_path(root, Path(remap_legacy_ceip_relpath(str(submap_rel))))

    paths: dict[str, Any] = {
        "cams_nc": str(cams.resolve()),
        "corine": str(corine.resolve()),
        "nuts_gpkg": str(nuts_gpkg.resolve()),
        "population_tif": str(pop_tif.resolve()),
        "osm_solvent_gpkg": str(osm_gpkg.resolve()),
        "ceip_workbook": str(ceip_workbook.resolve()),
        "ceip_subsector_map_yaml": str(submap_path.resolve()),
    }

    for opt in ("osm_roads_pbf", "osm_landuse_buildings_pbf"):
        v = paths_overlay.get(opt) or sp.get(opt)
        if v:
            p = resolve_path(root, Path(str(v)))
            if p.is_file():
                paths[opt] = str(p.resolve())

    ref_tif = sp.get("ref_tif", sector_cfg.get("ref_tif"))
    if ref_tif:
        p = resolve_path(root, Path(str(ref_tif)))
        if p.is_file():
            paths["ref_tif"] = str(p.resolve())

    sp_merge = {k: v for k, v in sp.items() if k not in ("paths", "ceip")}
    cfg = _deep_merge(cfg, sp_merge)
    if ceip_sp:
        cfg = _deep_merge(cfg, ceip_sp)
    cfg["paths"] = paths
    cfg["country"] = {"cams_iso3": iso3, "nuts_cntr": nuts_cntr}
    cfg["corine"] = {
        "band": int((sp.get("corine") or {}).get("band", 1)),
        "pad_m": float(sector_cfg.get("pad_m", 5000.0)),
    }
    cfg["output_dir"] = str(output_path.parent.resolve())
    cfg["output_tif"] = output_path.name
    cfg["tolerance_mass"] = float(cfg.get("tolerance_mass", 1e-5))
    if cfg.get("ceip_year") is not None and not cfg.get("ceip_years"):
        cfg["ceip_years"] = [int(cfg["ceip_year"])]
    return cfg


def run_solvents_pipeline(
    root: Path,
    cfg: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    from PROXY.sectors.E_Solvents.run_pipeline import run_solvents_area_pipeline

    return run_solvents_area_pipeline(root, cfg, config_path=config_path)


def run_e_solvents_pipeline(
    root: Path,
    cfg: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for older callers."""
    return run_solvents_pipeline(root, cfg, config_path=config_path)
