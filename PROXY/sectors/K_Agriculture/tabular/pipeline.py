"""Agriculture tabular model: class extent -> rho -> scores -> weights -> CSV."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import logging
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable, Iterator

import pandas as pd

from PROXY.core.dataloaders import resolve_path
from PROXY.sectors.K_Agriculture.k_config import load_agriculture_config, project_root
from PROXY.sectors.K_Agriculture.source_relevance import agricultural_soils, biomass_burning, enteric_grazing, fertilized_land
from PROXY.sectors.K_Agriculture.source_relevance import livestock_housing as livestock_housing_mod
from PROXY.sectors.K_Agriculture.source_relevance import manure, rice, soil_liming, urea_application
from PROXY.sectors.K_Agriculture.tabular.class_extent import build_class_extent_long, load_nuts2_filtered
from PROXY.sectors.K_Agriculture.tabular.scoring import compute_pollutant_score, merge_rho_lookup
from PROXY.sectors.K_Agriculture.tabular.weights import normalize_weights

logger = logging.getLogger(__name__)
ProcessFn = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]

PROCESS_REGISTRY: dict[str, ProcessFn] = {
    "enteric_grazing": enteric_grazing.compute_rho_df,
    "manure": manure.compute_rho_df,
    "fertilized_land": fertilized_land.compute_rho_df,
    "rice": rice.compute_rho_df,
    "biomass_burning": biomass_burning.compute_rho_df,
    "livestock_housing": livestock_housing_mod.compute_rho_df,
    "agricultural_soils": agricultural_soils.compute_rho_df,
    "soil_liming": soil_liming.compute_rho_df,
    "urea_application": urea_application.compute_rho_df,
}


@contextmanager
def _timed(label: str) -> Iterator[None]:
    t0 = perf_counter()
    try:
        yield
    finally:
        logger.info("K_Agriculture tabular timing: %s %.2fs", label, perf_counter() - t0)


def _alpha_pollutants(alpha_cfg: dict[str, Any]) -> list[str]:
    pol = alpha_cfg.get("pollutants") or {}
    return [str(key) for key in pol.keys()]


def _process_ids_needed(alpha_cfg: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    pol = alpha_cfg.get("pollutants") or {}
    for key in pol.keys():
        for row in pol.get(key) or []:
            pid = str(row.get("process_id", "")).strip()
            if pid:
                out.add(pid)
    return out


def _sanitize_token(name: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(name).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def _score_col(pollutant_key: str) -> str:
    return f"S_{_sanitize_token(pollutant_key)}"


def _weight_col(pollutant_key: str) -> str:
    return f"W_{_sanitize_token(pollutant_key)}"


def _per_pollutant_frame(
    extent_df: pd.DataFrame,
    rho_by_process: dict[str, pd.DataFrame],
    alpha_cfg: dict[str, Any],
    pollutant_key: str,
) -> pd.DataFrame:
    score_col = _score_col(pollutant_key)
    weight_col = _weight_col(pollutant_key)
    scored = compute_pollutant_score(extent_df, rho_by_process, alpha_cfg, pollutant_key, score_col)
    scored = normalize_weights(scored, score_col, weight_col)
    rho_diag = merge_rho_lookup(extent_df, rho_by_process, pollutant_key, alpha_cfg)[["NUTS_ID", "CLC_CODE", "rho_weighted"]]
    scored = scored.merge(rho_diag, on=["NUTS_ID", "CLC_CODE"], how="left")
    return scored


def _build_long_output(per_pollutant: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pollutant, df in per_pollutant.items():
        score_col = _score_col(pollutant)
        weight_col = _weight_col(pollutant)
        sub = df[["NUTS_ID", "CLC_CODE", "NAME_REGION", "COUNTRY", "n_pixels", score_col, weight_col, "rho_weighted"]].copy()
        sub = sub.rename(columns={score_col: "S_p", weight_col: "w_p"})
        sub.insert(0, "pollutant", pollutant)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(
            columns=["pollutant", "NUTS_ID", "CLC_CODE", "NAME_REGION", "COUNTRY", "n_pixels", "S_p", "w_p", "rho_weighted"]
        )
    return pd.concat(frames, ignore_index=True)


def _build_wide_output(long_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["NUTS_ID", "CLC_CODE", "NAME_REGION", "COUNTRY", "n_pixels"]
    keys = long_df[key_cols].drop_duplicates().copy()
    score_wide = long_df.pivot(index=["NUTS_ID", "CLC_CODE"], columns="pollutant", values="S_p")
    weight_wide = long_df.pivot(index=["NUTS_ID", "CLC_CODE"], columns="pollutant", values="w_p")
    rho_wide = long_df.pivot(index=["NUTS_ID", "CLC_CODE"], columns="pollutant", values="rho_weighted")
    score_wide.columns = [f"S_{_sanitize_token(c)}" for c in score_wide.columns]
    weight_wide.columns = [f"W_{_sanitize_token(c)}" for c in weight_wide.columns]
    rho_wide.columns = [f"RHO_{_sanitize_token(c)}" for c in rho_wide.columns]
    out = keys.merge(score_wide.reset_index(), on=["NUTS_ID", "CLC_CODE"], how="left")
    out = out.merge(weight_wide.reset_index(), on=["NUTS_ID", "CLC_CODE"], how="left")
    return out.merge(rho_wide.reset_index(), on=["NUTS_ID", "CLC_CODE"], how="left")


def apply_runtime_overrides(
    cfg: dict[str, Any],
    *,
    country_override: str | None = None,
    outputs_override_dir: Path | None = None,
    enable_visualization: bool | None = None,
    intermediary: bool | None = None,
) -> dict[str, Any]:
    runtime = copy.deepcopy(cfg)
    if country_override is not None:
        runtime.setdefault("run", {})
        runtime["run"]["country"] = country_override
    if outputs_override_dir is not None:
        runtime.setdefault("paths", {})
        runtime["paths"].setdefault("outputs", {})
        runtime["paths"]["output_dir"] = str(outputs_override_dir)
        runtime["paths"]["outputs"]["pollutant_dir"] = str(outputs_override_dir / "pollutants")
        runtime["paths"]["outputs"]["long_csv"] = str(outputs_override_dir / "weights_long.csv")
        runtime["paths"]["outputs"]["wide_csv"] = str(outputs_override_dir / "weights_wide.csv")
        runtime["paths"]["outputs"]["combined_csv"] = str(outputs_override_dir / "weights_wide.csv")
        runtime["paths"]["outputs"]["ch4_csv"] = str(outputs_override_dir / "weights_ch4.csv")
        runtime["paths"]["outputs"]["nox_csv"] = str(outputs_override_dir / "weights_nox.csv")
        runtime["paths"]["outputs"]["intermediary_dir"] = str(outputs_override_dir / "intermediary")
    if enable_visualization is not None:
        runtime.setdefault("visualization", {})
        runtime["visualization"]["enabled"] = enable_visualization
    if intermediary is not None:
        runtime.setdefault("run", {})
        runtime["run"]["intermediary"] = intermediary
    return runtime


def _build_rho_tables(extent_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    alpha_cfg = cfg.get("alpha") or {}
    needed = _process_ids_needed(alpha_cfg)
    logger.info("K_Agriculture tabular: needed process_ids=%s", sorted(needed))
    rho_by_process: dict[str, pd.DataFrame] = {}
    for pid in sorted(needed):
        fn = PROCESS_REGISTRY.get(pid)
        if fn is None:
            raise KeyError(f"No source_relevance module for process_id={pid!r}. Add to PROCESS_REGISTRY.")
        with _timed(f"rho {pid}"):
            rho_by_process[pid] = fn(extent_df, cfg)
        logger.info("K_Agriculture tabular: process=%s rho_rows=%d", pid, len(rho_by_process[pid]))
    return rho_by_process


def run_pipeline_with_config(cfg: dict[str, Any]) -> dict[str, Any]:
    root = project_root(cfg)
    run = cfg.get("run") or {}
    paths = cfg.get("paths") or {}
    outputs = paths.get("outputs") or {}
    inputs = paths.get("inputs") or {}

    nodata = float(run.get("nodata", -128.0))
    corine = resolve_path(
        root,
        inputs.get("corine_raster", "INPUT/Proxy/CORINE/U2018_CLC2018_V2020_20u1_100m.tif"),
    )

    with _timed("load NUTS2"):
        nuts2 = load_nuts2_filtered(cfg, root)
    logger.info("K_Agriculture tabular: NUTS2 rows=%d", len(nuts2))
    with _timed("class extent"):
        extent_df = build_class_extent_long(nuts2, corine, nodata=nodata)
    logger.info("K_Agriculture tabular: extent rows=%d", len(extent_df))

    alpha_cfg = cfg.get("alpha") or {}
    pollutant_keys = _alpha_pollutants(alpha_cfg)
    logger.info("K_Agriculture tabular: pollutants=%s", pollutant_keys)
    rho_by_process = _build_rho_tables(extent_df, cfg)
    with _timed("pollutant scores"):
        per_pollutant = {
            pollutant: _per_pollutant_frame(extent_df, rho_by_process, alpha_cfg, pollutant)
            for pollutant in pollutant_keys
        }

    long_df = _build_long_output(per_pollutant)
    wide_df = _build_wide_output(long_df)

    long_out = resolve_path(root, outputs.get("long_csv", "Agriculture/results/weights_long.csv"))
    wide_out = resolve_path(root, outputs.get("wide_csv", outputs.get("combined_csv", "Agriculture/results/weights_wide.csv")))
    ch4_out = resolve_path(root, outputs.get("ch4_csv", "Agriculture/results/nuts2_ch4_weights_by_clc.csv"))
    nox_out = resolve_path(root, outputs.get("nox_csv", "Agriculture/results/nuts2_nox_ag_weights_by_clc.csv"))
    pollutant_dir = resolve_path(root, outputs.get("pollutant_dir", "Agriculture/results/pollutants"))

    for p in (long_out, wide_out, ch4_out, nox_out):
        p.parent.mkdir(parents=True, exist_ok=True)
    pollutant_dir.mkdir(parents=True, exist_ok=True)

    with _timed("write CSVs"):
        long_df.to_csv(long_out, index=False, float_format="%.8f", na_rep="")
        wide_df.to_csv(wide_out, index=False, float_format="%.8f", na_rep="")
        if "CH4" in per_pollutant:
            per_pollutant["CH4"].to_csv(ch4_out, index=False, float_format="%.8f", na_rep="")
        if "NOx" in per_pollutant:
            per_pollutant["NOx"].to_csv(nox_out, index=False, float_format="%.8f", na_rep="")
        for pollutant, df in per_pollutant.items():
            out = pollutant_dir / f"{_sanitize_token(pollutant)}.csv"
            df.to_csv(out, index=False, float_format="%.8f", na_rep="")
    logger.info("K_Agriculture tabular: wrote long weights %s", long_out)
    logger.info("K_Agriculture tabular: wrote wide weights %s", wide_out)
    logger.info("K_Agriculture tabular: wrote per-pollutant CSVs under %s", pollutant_dir)

    run_cfg = cfg.get("run") or {}
    intermediary = bool(run_cfg.get("intermediary", outputs.get("intermediary", False)))
    if intermediary:
        inter_dir = resolve_path(root, outputs.get("intermediary_dir", "Agriculture/results/intermediary"))
        inter_dir.mkdir(parents=True, exist_ok=True)
        for pid, rdf in rho_by_process.items():
            fp = inter_dir / f"rho_{pid}.csv"
            rdf.to_csv(fp, index=False, float_format="%.8f", na_rep="")
        logger.info("K_Agriculture tabular: wrote intermediary rho tables under %s", inter_dir)

    vis = cfg.get("visualization") or {}
    if vis.get("enabled", False):
        try:
            from Agriculture.analysis.visualization.plot_nuts_clc_maps import run_visualization

            run_visualization(cfg, long_df)
        except ImportError:
            pass

    return {
        "long_csv": str(long_out),
        "wide_csv": str(wide_out),
        "pollutant_dir": str(pollutant_dir),
        "pollutants": pollutant_keys,
        "rows": len(long_df),
    }


def run_pipeline(config_path: Path | None = None, alpha_path: Path | None = None) -> dict[str, Any]:
    path = config_path
    if path is None:
        env = os.environ.get("AGRICULTURE_CONFIG", "").strip()
        path = Path(env) if env else None
    cfg = load_agriculture_config(path, alpha_path=alpha_path)
    return run_pipeline_with_config(cfg)
