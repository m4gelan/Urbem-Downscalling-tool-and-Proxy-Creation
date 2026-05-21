"""K_Agriculture public pipeline orchestration.

Merges path + sector configuration, loads CEIP alpha (method 1 / EU27 pool over seven
``family_*`` groups), and rasterizes the combined seven-family proxy stack onto the
CAMS K/L area-source grid.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np

from PROXY.core.alpha import default_ceip_profile_relpath, load_ceip_and_alpha
from PROXY.core.dataloaders import resolve_path
from PROXY.core.dataloaders.discovery import discover_cams_emissions, discover_corine
from PROXY.core.grid import reference_window_profile

from . import k_config
from .rasterize_kl import build_kl_sourcearea_tif

logger = logging.getLogger(__name__)


@contextmanager
def _timed(label: str) -> Iterator[None]:
    t0 = perf_counter()
    try:
        yield
    finally:
        logger.info("K_Agriculture timing: %s %.2fs", label, perf_counter() - t0)


def _pollutant_axes_from_sector(sector_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """CEIP internal keys (lowercase) and GeoTIFF band labels."""
    raw = sector_cfg.get("pollutants")
    default_keys = ["nh3", "nox", "nmvoc", "co", "pm10", "pm2_5"]
    default_labels = ["NH3", "NOx", "NMVOC", "CO", "PM10", "PM2.5"]
    if not raw:
        return default_keys, default_labels
    keys: list[str] = []
    labels: list[str] = []
    for p in raw:
        if isinstance(p, dict):
            tok = str(p.get("output", p.get("cams_var", ""))).strip()
        else:
            tok = str(p).strip()
        if not tok:
            continue
        pl = tok.lower().replace(".", "_")
        keys.append(pl)
        if pl == "pm2_5":
            labels.append("PM2.5")
        elif pl == "nh3":
            labels.append("NH3")
        elif pl == "nox":
            labels.append("NOx")
        elif pl == "nmvoc":
            labels.append("NMVOC")
        elif pl == "co":
            labels.append("CO")
        elif pl == "pm10":
            labels.append("PM10")
        else:
            labels.append(str(tok).strip().upper())
    return keys, labels


def merge_k_agriculture_pipeline_cfg(
    root: Path,
    path_cfg: dict[str, Any],
    sector_cfg: dict[str, Any],
    *,
    country: str,
) -> dict[str, Any]:
    """Merge path config, sector YAML, and agriculture defaults into one run config."""
    base = k_config.load_default_agriculture_dict()
    cfg: dict[str, Any] = deepcopy(base)
    pcommon = path_cfg.get("proxy_common") or {}
    psw = (path_cfg.get("proxy_specific") or {}).get("agriculture") or {}
    out_base = resolve_path(root, Path(sector_cfg["output_dir"])).resolve()
    tab_dir = out_base / "tabular"
    tab_dir.mkdir(parents=True, exist_ok=True)

    cams = discover_cams_emissions(
        root, Path(path_cfg.get("emissions", {}).get("cams_2019_nc", ""))
    )
    corine = discover_corine(root, Path(pcommon.get("corine_tif", "")))
    nuts_rel = pcommon.get("nuts_gpkg") or "Data/geometry/NUTS_RG_20M_2021_3035.gpkg"
    nuts = resolve_path(root, Path(nuts_rel))

    cfg["_project_root"] = root
    cfg.setdefault("run", {})
    cfg["run"]["country"] = str(country).strip().upper()
    cfg["run"].setdefault("year", 2020)
    cfg["run"].setdefault("nodata", -128.0)
    cfg["run"].setdefault(
        "intermediary", bool((sector_cfg.get("agriculture") or {}).get("write_intermediates", False))
    )
    if "nuts_cntr" in sector_cfg:
        cfg["run"]["country"] = str(sector_cfg["nuts_cntr"]).strip().upper()

    cfg["emissions_cams_path"] = str(cams.resolve())
    cfg["paths"] = cfg.get("paths") or {}
    cfg["paths"]["geometry"] = {"nuts_gpkg": str(nuts.resolve())}
    cfg["paths"].setdefault("census", {})
    cfg["paths"]["census"]["c21_gpkg"] = str(resolve_path(root, Path(psw.get("c21_gpkg", ""))))

    cdef = (base.get("paths") or {}).get("census") or {}
    for key in (
        "c21_field_map_json",
        "c21_explore_head_json",
        "enteric_factors_json",
        "housing_factors_json",
    ):
        rel = psw.get(key) or cdef.get(key) or (cfg.get("paths", {}).get("census", {}) or {}).get(key)
        if rel:
            p = Path(str(rel))
            cfg["paths"]["census"][key] = str(resolve_path(root, p) if not p.is_absolute() else p)

    cfg["paths"].setdefault("inputs", {})
    cfg["paths"]["inputs"]["corine_raster"] = str(corine.resolve())
    cfg["paths"].setdefault("outputs", {})
    cfg["paths"]["outputs"].update(
        {
            "intermediary_dir": str(tab_dir / "intermediary"),
        }
    )

    alpha_w = (
        (sector_cfg.get("ceip") or {}).get("workbook")
        or (sector_cfg.get("ceip") or {}).get("ceip_workbook")
        or pcommon.get("alpha_workbook")
        or "INPUT/Proxy/Alpha/Reported_Emissions_EU27_2018_2023.xlsx"
    )
    ceip_workbook = resolve_path(root, Path(str(alpha_w)))
    cfg["paths"]["ceip_workbook"] = str(ceip_workbook.resolve())
    gy_rel = default_ceip_profile_relpath(root, "K_Agriculture", "groups_yaml")
    ry_rel = default_ceip_profile_relpath(root, "K_Agriculture", "rules_yaml")
    cfg["paths"]["ceip_groups_yaml"] = str(resolve_path(root, Path(gy_rel)).resolve())
    cfg["paths"]["ceip_rules_yaml"] = str(resolve_path(root, Path(ry_rel)).resolve())
    cfg["paths"]["alpha_method_audit_dir"] = str(tab_dir.resolve())

    ceip_years = sector_cfg.get("ceip_years")
    if ceip_years is not None:
        cfg["paths"]["ceip_years"] = ceip_years

    raw_go = sector_cfg.get("ceip_group_order") or sector_cfg.get("group_order")
    if raw_go:
        cfg["group_order"] = tuple(str(x).strip() for x in raw_go if str(x).strip())
    else:
        cfg["group_order"] = tuple(f"family_{i}" for i in range(1, 8))

    pol_keys, pol_labels = _pollutant_axes_from_sector(sector_cfg)
    cfg["pollutants"] = pol_keys
    cfg["_raster_pollutant_labels"] = pol_labels
    cfg["ceip_pollutant_aliases"] = dict(sector_cfg.get("ceip_pollutant_aliases") or {})
    cfg["cntr_code_to_iso3"] = dict(sector_cfg.get("cntr_code_to_iso3") or {})

    cfg["lucas_build"] = cfg.get("lucas_build") or {}
    cfg["lucas_build"]["lucas_data"] = str(resolve_path(root, Path(psw["eu_lucas_2022_csv"])))
    snj = (
        psw.get("synthetic_n_rate")
        or psw.get("synthetic_n_rate_json")
        or "PROXY/config/agriculture/synthetic_N_rate.yaml"
    )
    cfg["lucas_build"]["synthetic_n_rate_json"] = str(resolve_path(root, Path(snj)))
    for key in (
        "gfed41s_agri_dm_mean_npy",
        "gfed41s_grid_area_npy",
        "gfed41s_nuts2_lookup_parquet",
    ):
        cfg["lucas_build"][key] = str(resolve_path(root, Path(psw[key])))

    lsoil = psw.get("lucas_soil_2018_csv")
    if not lsoil and psw.get("lucas_soil_2018_dir"):
        lsoil = str(Path(psw["lucas_soil_2018_dir"]) / "LUCAS-SOIL-2018.csv")
    if not lsoil:
        lsoil = "INPUT/Proxy/ProxySpecific/Agriculture/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv"
    cfg["lucas_build"]["lucas_soil_2018_csv"] = str(resolve_path(root, Path(lsoil)))
    cfg["lucas_build"]["output_csv"] = str(tab_dir / "lucas_nuts2_clc_relevance.csv")

    cfg["visualization"] = cfg.get("visualization") or {}
    cfg["visualization"].setdefault("enabled", False)
    cfg["visualization"].setdefault("output_dir", str(out_base / "plots"))
    cfg["_sector_cfg"] = deepcopy(sector_cfg)
    cfg["_path_cfg"] = deepcopy(path_cfg)
    cfg["_output_dir"] = str(out_base)

    logger.info("K_Agriculture paths: CAMS=%s", cams.resolve())
    logger.info("K_Agriculture paths: CORINE=%s", corine.resolve())
    logger.info("K_Agriculture paths: NUTS=%s", nuts.resolve())
    logger.info("K_Agriculture paths: C21=%s", cfg["paths"]["census"].get("c21_gpkg"))
    logger.info("K_Agriculture paths: LUCAS=%s", cfg["lucas_build"].get("lucas_data"))
    logger.info("K_Agriculture paths: LUCAS soil=%s", cfg["lucas_build"].get("lucas_soil_2018_csv"))
    logger.info("K_Agriculture paths: GFED dm=%s", cfg["lucas_build"].get("gfed41s_agri_dm_mean_npy"))
    logger.info("K_Agriculture paths: GFED area=%s", cfg["lucas_build"].get("gfed41s_grid_area_npy"))
    logger.info("K_Agriculture paths: GFED lookup=%s", cfg["lucas_build"].get("gfed41s_nuts2_lookup_parquet"))
    logger.info("K_Agriculture paths: CEIP workbook=%s", cfg["paths"]["ceip_workbook"])
    logger.info("K_Agriculture paths: CEIP groups=%s", cfg["paths"]["ceip_groups_yaml"])
    logger.info("K_Agriculture paths: CEIP rules=%s", cfg["paths"]["ceip_rules_yaml"])
    osm_block = path_cfg.get("osm") or {}
    ag_osm = osm_block.get("agriculture")
    if ag_osm:
        cfg.setdefault("paths", {}).setdefault("inputs", {})
        cfg["paths"]["inputs"]["agriculture_osm_gpkg"] = str(resolve_path(root, Path(str(ag_osm))).resolve())
        logger.info("K_Agriculture paths: OSM agriculture=%s", cfg["paths"]["inputs"]["agriculture_osm_gpkg"])
    if psw.get("koppen_raster_tif"):
        cfg.setdefault("paths", {}).setdefault("inputs", {})
        cfg["paths"]["inputs"]["koppen_raster_tif"] = str(
            resolve_path(root, Path(str(psw["koppen_raster_tif"]))).resolve()
        )
        logger.info("K_Agriculture paths: Köppen raster=%s", cfg["paths"]["inputs"]["koppen_raster_tif"])
    return cfg


def run_k_agriculture_pipeline(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Load CEIP alpha and rasterize the seven-family proxy stack to the K/L CAMS grid."""
    sector_cfg = cfg.get("_sector_cfg") or {}
    path_cfg = cfg.get("_path_cfg") or {}

    pcommon = path_cfg.get("proxy_common") or {}
    corine = discover_corine(root, Path(pcommon["corine_tif"]))
    nuts = resolve_path(root, Path(pcommon["nuts_gpkg"]))
    pad = float(sector_cfg.get("pad_m", 5000.0))
    nuts_for_ref = str((cfg.get("run") or {}).get("country") or sector_cfg.get("nuts_cntr") or "EL")
    with _timed("reference grid"):
        ref = reference_window_profile(
            corine_path=corine,
            nuts_gpkg=nuts,
            nuts_country=nuts_for_ref,
            pad_m=pad,
        )

    out_tif = sector_cfg.get("output_path")
    if out_tif is None:
        out_dir = resolve_path(root, Path(sector_cfg["output_dir"]))
        out_name = str(sector_cfg.get("output_filename", "agriculture_areasource.tif"))
        out_tif = (out_dir / out_name).resolve()
    else:
        out_tif = resolve_path(root, Path(out_tif)).resolve()

    cams = discover_cams_emissions(root, Path(path_cfg["emissions"]["cams_2019_nc"]))
    cams_iso3 = str(sector_cfg.get("cams_country_iso3", "GRC")).strip().upper()
    emi = sector_cfg.get("cams_emission_category_indices") or [14, 15]
    cams_emis = tuple(int(x) for x in emi)
    ag_clc = sector_cfg.get("ag_clc_codes")
    ag_t = tuple(int(x) for x in ag_clc) if ag_clc is not None else None
    corine_band = int(
        sector_cfg.get("corine_band", (path_cfg.get("corine") or {}).get("band", 1))
    )

    with _timed("CEIP alpha"):
        alpha, _fb, _wide = load_ceip_and_alpha(
            cfg,
            [cams_iso3],
            sector_key="K_Agriculture",
            focus_country_iso3=cams_iso3,
        )
    cfg["_ceip_alpha"] = np.asarray(alpha, dtype=np.float64)

    tab = {
        "long_csv": "",
        "wide_csv": "",
        "pollutant_dir": "",
        "pollutants": list(cfg.get("_raster_pollutant_labels") or cfg.get("pollutants") or []),
        "rows": 0,
    }

    with _timed("raster weights"):
        build_kl_sourcearea_tif(
            root,
            ref=ref,
            cams_nc=cams,
            nuts_gpkg=nuts,
            out_tif=out_tif,
            cams_iso3=cams_iso3,
            nuts_cntr=nuts_for_ref,
            pipeline_cfg=cfg,
            ag_clc_codes=ag_t,
            corine_band=corine_band,
            cams_emission_category_indices=cams_emis,
            run_validate=False,
        )

    logger.info("K_Agriculture pipeline complete: %s", out_tif)
    return {
        "output_path": str(out_tif),
        "weights_long": "",
        "tabular": tab,
    }
