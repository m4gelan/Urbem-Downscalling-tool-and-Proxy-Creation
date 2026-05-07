from __future__ import annotations

from pathlib import Path
from typing import Any

from PROXY.core.dataloaders import project_root
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
            "emep_ef": sector_config_dir / "EMEP_emission_factors.yaml",
            "gains_mapping": sector_config_dir / "GAINS_mapping.yaml",
            "eurostat_end_use_json": sector_config_dir / "eurostat_end_use.yaml",
            "hotmaps": {
                "heat_res": hm_dir / str(hm_names.get("heat_res", "heat_res_curr_density.tif")),
                "heat_nonres": hm_dir / str(hm_names.get("heat_nonres", "heat_nonres_curr_density.tif")),
                "gfa_res": hm_dir / str(hm_names.get("gfa_res", "gfa_res_curr_density.tif")),
                "gfa_nonres": hm_dir / str(hm_names.get("gfa_nonres", "gfa_nonres_curr_density.tif")),
            },
        },
        "corine": corine_block,
        "cams": sector_cfg.get("cams") or {},
        "base_proxy": sector_cfg["base_proxy"],
        "morphology": sector_cfg["morphology"],
        "gains": sector_cfg.get("gains") or {},
        "eurostat": sector_cfg.get("eurostat") or {},
        "co2": sector_cfg.get("co2") or {},
        "pollutants": sector_cfg.get("pollutants") or [],
        "run": sector_cfg.get("run") or {},
        "output_dir": sector_cfg.get("output_dir"),
    }

    output_path = Path(sector_cfg["output_path"])
    return run_other_combustion_weight_build(
        repo_root=root,
        cfg=merged,
        ref=ref,
        output_weights_path=output_path,
    )
