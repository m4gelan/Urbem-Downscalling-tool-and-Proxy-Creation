"""K_Agriculture public pipeline orchestration.

This module owns the sector-level flow: merge project/sector configuration, build the
NUTS2 x CLC tabular weights, then rasterize those weights onto the CAMS K/L area-source
grid. Source-relevance formulas remain in ``source_relevance`` and the tabular model.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from PROXY.core.dataloaders import resolve_path
from PROXY.core.dataloaders.discovery import discover_cams_emissions, discover_corine
from PROXY.core.grid import reference_window_profile

from . import k_config
from .tabular.pipeline import run_pipeline_with_config
from .rasterize_kl import build_kl_sourcearea_tif

logger = logging.getLogger(__name__)


@contextmanager
def _timed(label: str) -> Iterator[None]:
    t0 = perf_counter()
    try:
        yield
    finally:
        logger.info("K_Agriculture timing: %s %.2fs", label, perf_counter() - t0)


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
            "long_csv": str(tab_dir / "weights_long.csv"),
            "wide_csv": str(tab_dir / "weights_wide.csv"),
            "pollutant_dir": str(tab_dir / "pollutants"),
            "combined_csv": str(tab_dir / "weights_wide.csv"),
            "ch4_csv": str(tab_dir / "nuts2_ch4_weights_by_clc.csv"),
            "nox_csv": str(tab_dir / "nuts2_nox_ag_weights_by_clc.csv"),
            "intermediary_dir": str(tab_dir / "intermediary"),
        }
    )

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

    alpha_yaml = root / "PROXY" / "config" / "agriculture" / "alpha.config.yaml"
    alpha_json = root / "PROXY" / "config" / "agriculture" / "alpha.config.json"
    cfg["alpha"] = k_config.load_alpha_config(alpha_yaml if alpha_yaml.is_file() else alpha_json)

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
    return cfg


def run_k_agriculture_pipeline(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Run tabular K_Agriculture weights and rasterize them to the K/L CAMS grid."""
    sector_cfg = cfg.get("_sector_cfg") or {}
    path_cfg = cfg.get("_path_cfg") or {}

    with _timed("tabular weights"):
        tab = run_pipeline_with_config(cfg)
    weights_long = Path(tab["long_csv"])

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
    logger.info(
        "K_Agriculture reference grid: crs=%s width=%s height=%s",
        ref.get("crs"),
        ref.get("width"),
        ref.get("height"),
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
    pols = sector_cfg.get("raster_pollutants")
    if not pols:
        pols = list((cfg.get("alpha") or {}).get("pollutants", {}).keys())
    if not pols:
        pols = ["CH4", "NH3", "NOx"]
    ag_clc = sector_cfg.get("ag_clc_codes")
    ag_t = tuple(int(x) for x in ag_clc) if ag_clc is not None else None
    corine_band = int(
        sector_cfg.get("corine_band", (path_cfg.get("corine") or {}).get("band", 1))
    )

    with _timed("raster weights"):
        build_kl_sourcearea_tif(
            root,
            ref=ref,
            cams_nc=cams,
            weights_long=weights_long,
            nuts_gpkg=nuts,
            out_tif=out_tif,
            cams_iso3=cams_iso3,
            nuts_cntr=nuts_for_ref,
            pollutants=[str(p) for p in pols],
            ag_clc_codes=ag_t,
            corine_band=corine_band,
            cams_emission_category_indices=cams_emis,
            run_validate=False,
        )

    logger.info("K_Agriculture pipeline complete: %s", out_tif)
    return {
        "output_path": str(out_tif),
        "weights_long": str(weights_long),
        "tabular": tab,
    }
