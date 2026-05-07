"""Merge PROXY path + sector config for G_Shipping (no CEIP: single shared spatial proxy)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from PROXY.core.dataloaders import resolve_path
from PROXY.core.dataloaders.discovery import discover_cams_emissions, discover_corine
from PROXY.core.grid import resolve_nuts_cntr_code
from PROXY.core.ref_profile import load_area_ref_profile


def _load_shipping_base(root: Path) -> dict[str, Any]:
    candidates = [
        root / "PROXY" / "config" / "ceip" / "profiles" / "shipping_pipeline.yaml",
        root / "PROXY" / "config" / "shipping" / "defaults.json",
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
        "G_Shipping base config not found: expected PROXY/config/ceip/profiles/shipping_pipeline.yaml "
        "or legacy PROXY/config/shipping/defaults.json"
    )


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def merge_shipping_pipeline_cfg(
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
    *,
    country: str,
    output_dir: Path,
) -> dict[str, Any]:
    """
    Build cfg for :func:`PROXY.sectors.G_Shipping.pipeline.run_shipping_area_pipeline`.
    All pollutants share the same 1-band spatial proxy; no per-pollutant alpha.
    """
    cfg = deepcopy(_load_shipping_base(root))
    sov = sector_cfg.get("shipping") or {}
    pcommon = path_cfg.get("proxy_common") or {}
    pss = (path_cfg.get("proxy_specific") or {}).get("shipping") or {}

    emod = pss.get("vessel_density_tif")
    if not emod or not str(emod).strip():
        raise KeyError("Set proxy_specific.shipping.vessel_density_tif in paths.yaml")

    cams_nc = discover_cams_emissions(
        root, resolve_path(root, Path(path_cfg["emissions"]["cams_2019_nc"]))
    )
    corine = discover_corine(
        root, resolve_path(root, Path(pcommon["corine_tif"]))
    )
    nuts_gpkg = resolve_path(root, pcommon["nuts_gpkg"])
    emodnet = resolve_path(root, Path(str(emod)))
    osm = resolve_path(root, path_cfg["osm"]["shipping"])

    if not emodnet.is_file():
        raise FileNotFoundError(f"EMODnet vessel density not found: {emodnet}")

    iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()
    nuts_override = str(sector_cfg.get("nuts_cntr", "")).strip().upper()
    nuts_cntr = nuts_override if len(nuts_override) == 2 else resolve_nuts_cntr_code(country)
    try:
        gnfr_g = int((sector_cfg.get("cams_emission_category_indices") or [10])[0])
    except (TypeError, ValueError):
        gnfr_g = 10

    out_tif = str(sov.get("output_filename") or sector_cfg.get("output_filename") or "shipping_areasource.tif")

    base_proxy = (cfg.get("proxy") or {}).copy()
    if isinstance(sov.get("proxy"), dict):
        base_proxy = {**base_proxy, **sov["proxy"]}
    wdiag = bool(
        sov.get("write_diagnostics", base_proxy.get("write_diagnostics", False))
    )
    if "write_diagnostics" in (sov.get("output") or {}):
        wdiag = bool(sov["output"].get("write_diagnostics", wdiag))

    paths: dict[str, Any] = {
        "cams_nc": str(cams_nc.resolve()),
        "corine": str(corine.resolve()),
        "nuts_gpkg": str(nuts_gpkg.resolve()),
        "emodnet": str(emodnet.resolve()),
        "osm_gpkg": str(osm.resolve()),
    }
    for key in ("ref_tif",):
        v = sov.get(key) or sector_cfg.get(key)
        if v and str(v).strip():
            rp = resolve_path(root, Path(str(v)))
            if rp.is_file():
                paths["ref_tif"] = str(rp.resolve())
    p_overlay = sov.get("paths") if isinstance(sov.get("paths"), dict) else None
    if p_overlay:
        for key, val in p_overlay.items():
            if val is None or str(val).strip() == "":
                continue
            if key in ("cams_nc", "corine", "nuts_gpkg", "emodnet", "osm_gpkg"):
                paths[key] = str(resolve_path(root, Path(str(val))).resolve())

    cfg["paths"] = paths
    cfg["corine_window"] = {
        "nuts_cntr": nuts_cntr,
        "pad_m": float(sector_cfg.get("pad_m", 5000.0)),
    }
    cfg["cams"] = _deep_merge(cfg.get("cams") or {}, {"gnfr_g_index": gnfr_g, "country_iso3": iso3})
    cfg["output"] = {
        "dir": str(output_dir.resolve()),
        "weights_tif": out_tif,
        "write_diagnostics": wdiag,
    }
    cfg["proxy"] = {**base_proxy, "write_diagnostics": wdiag}
    if isinstance(sov.get("corine_window"), dict):
        cfg["corine_window"] = {**cfg.get("corine_window", {}), **sov["corine_window"]}
    if isinstance(sov.get("cams"), dict):
        cfg["cams"] = {**cfg.get("cams", {}), **sov["cams"]}
    if isinstance(sov.get("logging"), dict):
        cfg["logging"] = {**cfg.get("logging", {}), **sov["logging"]}
    return cfg


def run_shipping_area_pipeline(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Build reference grid, combined proxy, CAMS-cell normalize; single-band GeoTIFF."""
    from PROXY.sectors.G_Shipping.proxy_shipping import run_shipping_areasource

    full: dict[str, Any] = {**cfg, "_project_root": root}
    ref = load_area_ref_profile(full)
    paths = full["paths"]
    pr = full.get("proxy") or {}
    out_dir = Path(str(full["output"]["dir"])).resolve()
    out_name = str(full["output"].get("weights_tif", "shipping_areasource.tif"))
    wdiag = bool(
        full["output"].get("write_diagnostics", pr.get("write_diagnostics", False))
    )

    res = run_shipping_areasource(
        cams_nc=Path(paths["cams_nc"]),
        corine_path=None,
        osm_gpkg=Path(paths["osm_gpkg"]),
        emodnet_path=Path(paths["emodnet"]),
        output_folder=out_dir,
        ref=ref,
        output_filename=out_name,
        write_diagnostics=wdiag,
        osm_subdivide=int(pr.get("osm_subdivide", 4)),
        land_damp=float(pr.get("land_damp", 0.12)),
        w_emodnet=float(pr.get("weight_emodnet", 0.25)),
        w_osm=float(pr.get("weight_osm", 0.5)),
        w_port=float(pr.get("weight_port", 0.25)),
        port_level2=int(pr.get("port_level2", 123)),
    )
    return {
        "output_path": res["output_tif"],
        "output_dir": str(out_dir),
        "validate_errors": res.get("validate_errors") or [],
    }
