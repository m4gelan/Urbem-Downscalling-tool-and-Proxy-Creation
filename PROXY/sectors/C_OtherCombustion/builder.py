from __future__ import annotations

from pathlib import Path
from typing import Any

from PROXY.core.alpha.ceip_index_loader import default_ceip_profile_relpath, remap_legacy_ceip_relpath
from PROXY.core.dataloaders import project_root, resolve_path
from PROXY.core.grid import reference_window_profile
from PROXY.sectors.C_OtherCombustion.pipeline import run_other_combustion_weight_build


def build(*, path_cfg: dict[str, Any], sector_cfg: dict[str, Any], country: str) -> dict[str, Any]:
    """
    GNFR C other combustion: merge paths and science config, build the CORINE reference window,
    then :func:`PROXY.sectors.C_OtherCombustion.pipeline.run_other_combustion_weight_build` for multiband weights.
    """
    root = project_root()
    spec = (path_cfg.get("proxy_specific") or {}).get("other_combustion") or {}
    if not spec:
        raise ValueError(
            "paths.yaml must define proxy_specific.other_combustion (gains_dir, hotmaps_dir). "
            "Eurostat uses the API + PROXY/cache/eurostat when eurostat.enabled is true."
        )

    sector_config_dir = root / "PROXY" / "config" / "other_combustion"
    for name in ("EMEP_emission_factors.yaml", "GAINS_mapping.yaml", "eurostat_end_use.yaml"):
        p = sector_config_dir / name
        if not p.is_file():
            raise FileNotFoundError(f"Missing sector science config: {p}")

    corine_path = Path(path_cfg["proxy_common"]["corine_tif"])
    nuts_gpkg = Path(path_cfg["proxy_common"]["nuts_gpkg"])
    corine_block = sector_cfg.get("corine") or {}
    pad_m = float(corine_block.get("pad_m", sector_cfg.get("pad_m", 5000.0)))

    ref = reference_window_profile(
        corine_path=corine_path,
        nuts_gpkg=nuts_gpkg,
        nuts_country=country.strip().upper(),
        pad_m=pad_m,
    )

    hm_dir = Path(spec["hotmaps_dir"])
    hm_names = sector_cfg.get("hotmaps") or {}
    proxy_common = path_cfg.get("proxy_common") or {}
    alpha_w = proxy_common.get("alpha_workbook")
    if not alpha_w:
        raise ValueError(
            "paths.yaml proxy_common.alpha_workbook is required for GNFR C (CEIP reported emissions workbook)."
        )
    path_osm = path_cfg.get("osm") or {}
    oc_rel = path_osm.get("other_combustion") or path_osm.get("industry")
    if not oc_rel:
        raise ValueError(
            "paths.yaml must define osm.other_combustion (or osm.industry as fallback) for GNFR C off-road OSM mask."
        )

    ceip_ov = sector_cfg.get("ceip") if isinstance(sector_cfg.get("ceip"), dict) else {}
    gy_rel = remap_legacy_ceip_relpath(
        str(ceip_ov.get("groups_yaml") or default_ceip_profile_relpath(root, "C_OtherCombustion", "groups_yaml"))
    )
    ry_rel = remap_legacy_ceip_relpath(
        str(ceip_ov.get("rules_yaml") or default_ceip_profile_relpath(root, "C_OtherCombustion", "rules_yaml"))
    )

    waste_spec = (path_cfg.get("proxy_specific") or {}).get("waste") or {}
    merged: dict[str, Any] = {
        "country": {
            "cams_iso3": str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper(),
            "nuts_cntr": country.strip().upper(),
        },
        "paths": {
            "cams_nc": path_cfg["emissions"]["cams_2019_nc"],
            "nuts_gpkg": nuts_gpkg,
            "gains_dir": spec["gains_dir"],
            "eurostat_xlsx": spec.get("eurostat_xlsx"),
            "population_tif": proxy_common.get("population_tif"),
            "ghsl_smod_tif": waste_spec.get("ghsl_smod_tif"),
            "ceip_workbook": resolve_path(root, Path(alpha_w)),
            "ceip_groups_yaml": resolve_path(root, Path(gy_rel)),
            "ceip_rules_yaml": resolve_path(root, Path(ry_rel)),
            "osm_other_combustion_gpkg": resolve_path(root, Path(oc_rel)),
            "emep_ef": sector_config_dir / "EMEP_emission_factors.yaml",
            "gains_mapping": sector_config_dir / "GAINS_mapping.yaml",
            "eurostat_end_use_json": sector_config_dir / "eurostat_end_use.yaml",
            "hotmaps": {
                "heat_res": hm_dir / str(hm_names.get("heat_res", "heat_res_curr_density.tif")),
                "heat_nonres": hm_dir / str(hm_names.get("heat_nonres", "heat_nonres_curr_density.tif")),
                "hdd_curr": hm_dir / str(hm_names.get("hdd_curr", "HDD_curr.tif")),
                "gfa_res": hm_dir / str(hm_names.get("gfa_res", "gfa_res_curr_density.tif")),
                "gfa_nonres": hm_dir / str(hm_names.get("gfa_nonres", "gfa_nonres_curr_density.tif")),
            },
        },
        "appliance_proxy": sector_cfg.get("appliance_proxy") or {},
        "corine": corine_block,
        "cams": sector_cfg.get("cams") or {},
        "base_proxy": sector_cfg["base_proxy"],
        "morphology": sector_cfg["morphology"],
        "gains": sector_cfg.get("gains") or {},
        "eurostat": sector_cfg.get("eurostat") or {},
        "co2": sector_cfg.get("co2") or {},
        "pollutants": sector_cfg.get("pollutants") or [],
        "run": sector_cfg.get("run") or {},
        "ceip": {
            "years": ceip_ov.get("years"),
            "cntr_code_to_iso3": ceip_ov.get("cntr_code_to_iso3") or {},
            "pollutant_aliases": ceip_ov.get("pollutant_aliases") or {},
        },
        "output_dir": sector_cfg.get("output_dir"),
    }

    output_path = Path(sector_cfg["output_path"])
    return run_other_combustion_weight_build(
        repo_root=root,
        cfg=merged,
        ref=ref,
        output_weights_path=output_path,
    )
