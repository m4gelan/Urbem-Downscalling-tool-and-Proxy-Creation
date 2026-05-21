from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from PROXY_V2.core import log
from PROXY_V2.core.alias import gamma_split_manure
from PROXY_V2.dataset_loaders.load_corine import load_corine, load_corine_weighted_l3
from PROXY_V2.dataset_loaders.load_lucas import load_lucas_points, lucas_rate_mean_raster
from PROXY_V2.dataset_loaders.load_nuts2 import load_nuts2_polygons, rasterize_nuts2_ids
from PROXY_V2.dataset_loaders.load_cams_cells_mask import pixels_inside_cams_cells


@dataclass(frozen=True)
class AgReferenceGrid:
    height: int
    width: int
    transform: Any
    crs: Any
    cell_id: np.ndarray
    nuts_r: np.ndarray
    nuts_id_to_idx: dict[str, int]
    cams_cells: dict[int, dict[str, Any]]
    cams_grid: dict[str, Any]


def build_ag_reference_grid(
    repo_root: Path,
    *,
    corine_path: Path,
    corine_band: int,
    corine_l3_anchor: list[int],
    nuts_path: Path,
    country_profile: dict[str, str],
    cams_cells: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
) -> AgReferenceGrid:
    cor_path = repo_root / str(corine_path).replace("\\", "/")
    nuts_path_r = repo_root / str(nuts_path).replace("\\", "/")

    _, cor_tr, cor_crs, cell_id = load_corine(
        cor_path,
        [int(x) for x in corine_l3_anchor],
        int(corine_band),
        cams_cells,
        cams_grid,
    )
    h, w = cell_id.shape
    log.info(f"K_Agriculture reference grid: {h} x {w} px")

    nuts_gdf = load_nuts2_polygons(nuts_path_r, country_profile)
    nuts_r, nuts_id_to_idx = rasterize_nuts2_ids(nuts_gdf, h, w, cor_tr, cor_crs)

    return AgReferenceGrid(
        height=h,
        width=w,
        transform=cor_tr,
        crs=cor_crs,
        cell_id=cell_id,
        nuts_r=nuts_r,
        nuts_id_to_idx=nuts_id_to_idx,
        cams_cells=cams_cells,
        cams_grid=cams_grid,
    )


def pixel_area_ha(transform: Any, height: int, width: int) -> np.ndarray:
    ha = abs(float(transform.a) * float(transform.e)) / 10000.0
    return np.full((int(height), int(width)), ha, dtype=np.float32)


def crop_groups_from_lucas_cfg(lucas_cfg: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    groups = [str(x) for x in lucas_cfg["lucas_crop_group_priority"]]
    return groups, {g: i + 1 for i, g in enumerate(groups)}


def load_housing_factor_params(ef_path: Path) -> dict[str, dict[str, float]]:
    with ef_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    block = (doc or {}).get("housing_factor")
    if not isinstance(block, dict):
        raise ValueError(f"housing_factor block missing in {ef_path}")
    out: dict[str, dict[str, float]] = {}
    for species, spec in block.items():
        if str(species).startswith("_") or not isinstance(spec, dict):
            continue
        out[str(species).strip()] = {
            "housing_fraction": float(spec["housing_fraction"]),
            "N_excretion": float(spec["N_excretion"]),
            "liveweight_kg": float(spec["liveweight_kg"]),
        }
    log.debug(f"housing_factor species keys: {sorted(out.keys())}")
    return out


def load_manure_application_params(ef_path: Path) -> dict[str, float | dict[str, float]]:
    with ef_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    block = (doc or {}).get("manure_application_to_soils")
    if not isinstance(block, dict):
        raise ValueError(f"manure_application_to_soils block missing in {ef_path}")
    ratios = block.get("nonfodder_rate_ratios")
    if not isinstance(ratios, dict):
        raise ValueError("manure_application_to_soils.nonfodder_rate_ratios required")
    return {
        "storage_n_loss_fraction": float(block["storage_n_loss_fraction"]),
        "nonfodder_rate_ratios": {str(k): float(v) for k, v in ratios.items()},
    }


def _species_n_applied(
    heads: float,
    params: dict[str, float],
    storage_loss: float,
) -> float:
    if heads <= 0:
        return 0.0
    return (
        heads
        * params["liveweight_kg"]
        * params["N_excretion"]
        * (365.0 / 1000.0)
        * params["housing_fraction"]
        * (1.0 - storage_loss)
    )


def build_manure_nuts_pools(
    headcounts_df: pd.DataFrame,
    housing_params: dict[str, dict[str, float]],
    storage_n_loss_fraction: float,
    country_profile: dict[str, str],
    *,
    country_nuts_ids: set[str] | None = None,
) -> dict[str, dict[str, float]]:
    gamma = gamma_split_manure(country_profile)
    L = float(storage_n_loss_fraction)
    out: dict[str, dict[str, float]] = {}

    for nuts_id, row in headcounts_df.iterrows():
        nid = str(nuts_id).strip().upper()
        by_sp: dict[str, float] = {}
        for col in headcounts_df.columns:
            sp = str(col).strip().lower()
            if sp not in housing_params:
                raise KeyError(f"species {col!r} missing from housing_factor")
            by_sp[sp] = _species_n_applied(float(row[col]), housing_params[sp], L)

        n_pigs = by_sp.get("pigs", 0.0)
        n_fodder = (
            by_sp.get("bovine", 0.0)
            + by_sp.get("sheep", 0.0)
            + by_sp.get("goats", 0.0)
            + gamma * n_pigs
        )
        n_non = (1.0 - gamma) * n_pigs + by_sp.get("poultry", 0.0)
        out[nid] = {
            "N_fodder": n_fodder,
            "N_nonfodder": n_non,
            "N_total": n_fodder + n_non,
            **{f"N_{k}": v for k, v in by_sp.items()},
        }
        if country_nuts_ids is None or nid in country_nuts_ids:
            log.debug(
                f"manure-N {nid}: fodder={n_fodder:.2f} nonfodder={n_non:.2f} gamma={gamma:.2f}"
            )

    if country_nuts_ids:
        sub = {k: v for k, v in out.items() if k in country_nuts_ids}
        tot_f = sum(v["N_fodder"] for v in sub.values())
        tot_n = sum(v["N_nonfodder"] for v in sub.values())
        log.info(
            f"manure-N pools ({len(sub)} NUTS2): gamma={gamma:.2f} "
            f"N_fodder={tot_f:.2e} N_nonfodder={tot_n:.2e} kg/yr"
        )
    return out


def load_nmvoc_lc1_ef(ef_path: Path) -> dict[str, float]:
    with ef_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    block = (doc or {}).get("nmvoc_crop_ef")
    if not isinstance(block, dict):
        raise ValueError(f"nmvoc_crop_ef block missing in {ef_path}")
    rates = block.get("lc1_ef")
    if not isinstance(rates, dict):
        raise ValueError("nmvoc_crop_ef.lc1_ef required")
    return {str(k).strip().upper(): float(v) for k, v in rates.items()}


def load_einarsson_lc1_rates(ef_path: Path) -> dict[str, float]:
    with ef_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    block = (doc or {}).get("inorganic_n_fertilizer")
    if not isinstance(block, dict):
        raise ValueError(f"inorganic_n_fertilizer block missing in {ef_path}")
    rates = block.get("lc1_rates_kg_n_ha_yr")
    if not isinstance(rates, dict):
        raise ValueError("inorganic_n_fertilizer.lc1_rates_kg_n_ha_yr required")
    return {str(k).strip().upper(): float(v) for k, v in rates.items()}


def _norm_lc1(cell: Any) -> str:
    s = str(cell).strip().strip('"').strip("'").upper()
    return " ".join(s.split())


def eligible_synthetic_n_lc1(lc1: Any, lu1: Any) -> bool:
    lc = _norm_lc1(lc1)
    lu = _norm_lc1(lu1)
    if not lc:
        return False
    if lc.startswith("B"):
        return True
    if lc == "E10":
        return True
    return lc == "E20" and lu == "U111"


def _weighted_lucas_rate(sub: pd.DataFrame, lc_codes: set[str]) -> tuple[float, int]:
    if sub.empty:
        return 0.0, 0
    s = sub[sub["lc1"].isin(lc_codes)]
    if s.empty:
        return 0.0, 0
    return float(s["rate"].sum()) / len(s), len(s)


def build_clc_fallback_lookup(
    pts: pd.DataFrame,
    clc_map: dict[int, list[str]],
    ag_l3_codes: list[int],
    nuts_id_to_idx: dict[str, int],
    country_nuts: set[str],
    lc1_rates: dict[str, float],
    n_min: int,
    *,
    collect_debug: bool = False,
) -> tuple[np.ndarray, dict[int, int], list[Any]]:
    from PROXY_V2.writers.debug_dump import ClcRateCell

    n_nuts = len(nuts_id_to_idx) + 1
    l3_codes = [int(x) for x in ag_l3_codes]
    l3_to_col = {l3: i for i, l3 in enumerate(l3_codes)}
    lookup = np.zeros((n_nuts, len(l3_codes)), dtype=np.float32)
    debug_cells: list[ClcRateCell] = []

    for l3 in l3_codes:
        col = l3_to_col[l3]
        codes = {_norm_lc1(c) for c in clc_map[int(l3)]}
        eu_vals = [lc1_rates[c] for c in codes if c in lc1_rates]
        eu_mean = float(np.mean(eu_vals)) if eu_vals else 0.0

        for nid, idx in nuts_id_to_idx.items():
            r, n = _weighted_lucas_rate(pts[pts["nuts2"] == nid], codes)
            if n >= n_min:
                lookup[idx, col] = r
                src = "nuts2"
            else:
                r, n = _weighted_lucas_rate(pts[pts["nuts2"].isin(country_nuts)], codes)
                if n >= n_min:
                    lookup[idx, col] = float(r)
                    src = "country"
                else:
                    lookup[idx, col] = eu_mean
                    src = "eu_mean"
                    n = 0
            if collect_debug:
                debug_cells.append(
                    ClcRateCell(
                        nuts2=str(nid).strip().upper(),
                        l3=int(l3),
                        rate=float(lookup[idx, col]),
                        n_lucas=int(n),
                        source=src,
                        eu_mean=eu_mean,
                    )
                )

    return lookup, l3_to_col, debug_cells


def lucas_lc1_rate_points(
    lucas_path: Path,
    country_profile: dict[str, str],
    lucas_cfg: dict[str, Any],
    signal_cfg: dict[str, Any],
    lc1_rates: dict[str, float],
    nuts2_col: str,
    *,
    broad_arable: bool,
) -> pd.DataFrame:
    from PROXY_V2.dataset_loaders.load_lucas import load_lucas_points

    pts = load_lucas_points(
        lucas_path,
        country_profile,
        signal_cfg["lucas_filter"],
        lucas_root_cfg=lucas_cfg,
        extra_columns=(nuts2_col, "SURVEY_LU1", "SURVEY_LC1"),
    )
    lc = pts["SURVEY_LC1"].map(_norm_lc1)
    lu = pts["SURVEY_LU1"].map(_norm_lc1)
    if broad_arable:
        keep = (
            lc.str.startswith("B")
            | (lc == "E10")
            | ((lc == "E20") & (lu == "U111"))
        ) & lc.isin(lc1_rates)
    else:
        keep = lc.isin(lc1_rates)
        if "E20" in lc1_rates:
            keep &= (lc != "E20") | (lu == "U111")
    pts = pts.loc[keep].copy()
    pts["lc1"] = lc.loc[keep].values
    pts["nuts2"] = pts[nuts2_col].astype(str).str.strip().str.upper()
    pts["rate"] = pts["lc1"].map(lc1_rates).astype(np.float64)
    log.info(f"LUCAS LC1-rate points: {len(pts)}")
    return pts[["lat", "lon", "nuts2", "lc1", "rate"]]


def build_lc1_rate_surface(
    repo_root: Path,
    cfg: dict[str, Any],
    country_profile: dict[str, str],
    *,
    ref: AgReferenceGrid,
    cams_cells: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
    corine_filepath: str | Path,
    lucas_filepath: str | Path,
    lucas_block_key: str,
    lc1_rates: dict[str, float],
    broad_arable: bool,
    log_label: str,
    collect_debug: bool = False,
    debug_title: str = "",
    debug_units: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any | None]:
    from PROXY_V2.writers.debug_dump import Lc1RateSignalDebug
    lucas_cfg = cfg.get("LUCAS") or {}
    corine_cfg = cfg.get("corine") or {}
    osm_cfg = cfg.get("osm") or {}
    syn_cfg = lucas_cfg.get(lucas_block_key) or {}
    corine_band = int(corine_cfg["band"])
    ag_l3 = [int(x) for x in syn_cfg["agricultural_l3_codes"]]
    buffer_m = float(syn_cfg["lucas_buffer_m"])
    n_min = int(syn_cfg["nuts2_min_lucas_count"])
    nuts2_col = str(lucas_cfg.get("nuts2_column", "POINT_NUTS2"))
    clc_map = {int(k): [str(c) for c in v] for k, v in syn_cfg["clc_to_lucas_lc1"].items()}
    country_nuts = {str(k).strip().upper() for k in ref.nuts_id_to_idx}

    pts = lucas_lc1_rate_points(
        repo_root / str(lucas_filepath).replace("\\", "/"),
        country_profile,
        lucas_cfg,
        syn_cfg,
        lc1_rates,
        nuts2_col,
        broad_arable=broad_arable,
    )
    pts = pts[pts["nuts2"].isin(country_nuts)]

    from PROXY_V2.dataset_loaders.load_corine import load_corine
    from PROXY_V2.dataset_loaders.load_lucas import lucas_rate_mean_raster

    r_lucas = lucas_rate_mean_raster(
        pts,
        buffer_m,
        str(osm_cfg["metric_crs"]),
        height=ref.height,
        width=ref.width,
        transform=ref.transform,
        raster_crs=ref.crs,
    )
    lookup, l3_to_col, debug_cells = build_clc_fallback_lookup(
        pts,
        clc_map,
        ag_l3,
        ref.nuts_id_to_idx,
        country_nuts,
        lc1_rates,
        n_min,
        collect_debug=collect_debug,
    )
    l3_r, _, _, _ = load_corine(
        repo_root / str(corine_filepath).replace("\\", "/"),
        ag_l3,
        corine_band,
        cams_cells,
        cams_grid,
        need_cell_id=False,
        return_l3=True,
    )
    l3_r = np.asarray(l3_r, dtype=np.int16)

    ag = l3_r > 0
    ni = np.clip(ref.nuts_r.astype(np.int32), 0, lookup.shape[0] - 1)
    r_clc = np.zeros(l3_r.shape, dtype=np.float32)
    for l3, ci in l3_to_col.items():
        m = l3_r == l3
        if m.any():
            r_clc[m] = lookup[ni[m], ci]
    r = np.where(np.isfinite(r_lucas), r_lucas, np.where(ag, r_clc, 0.0)).astype(np.float32)

    area_ha = pixel_area_ha(ref.transform, ref.height, ref.width)
    s = (area_ha * r).astype(np.float32)
    inside = (ref.nuts_r > 0) & ag
    mx = float(s[inside].max()) if inside.any() else 0.0
    norm = np.zeros_like(s)
    if mx > 0:
        norm[inside] = (s[inside] / mx).astype(np.float32)

    n_lucas = int(np.isfinite(r_lucas[inside]).sum())
    n_clc = int((ag & ~np.isfinite(r_lucas)).sum())
    log.info(
        f"{log_label}: R max={float(r.max()):.4g} S sum={float(s.sum()):.4g} "
        f"px LUCAS={n_lucas} CLC-fallback={n_clc} norm max={float(norm.max()):.4g}"
    )
    dbg = None
    if collect_debug:
        dbg = Lc1RateSignalDebug(
            title=debug_title or log_label,
            units=debug_units,
            l3_codes=ag_l3,
            lc1_ef=dict(lc1_rates),
            cells=debug_cells,
            n_lucas_points=len(pts),
        )
    return r, s, norm, dbg


def corine_cropland_mask_on_ref(
    repo_root: Path,
    corine_path: Path,
    l3_codes: list[int],
    corine_band: int,
    cams_cells: dict[int, dict[str, Any]],
    *,
    ref_height: int,
    ref_width: int,
    ref_transform: Any,
    ref_crs: Any,
) -> np.ndarray:
    """Binary 0/1 cropland mask on the agriculture reference grid."""
    weights = {int(l3): 1.0 for l3 in l3_codes}
    acc = load_corine_weighted_l3(
        repo_root / str(corine_path).replace("\\", "/"),
        weights,
        int(corine_band),
        cams_cells,
        ref_height=int(ref_height),
        ref_width=int(ref_width),
        ref_transform=ref_transform,
        ref_crs=ref_crs,
    )
    return (acc > 0).astype(np.float32)


def viirs_season(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def z_score_by_cams_cell(
    frp: np.ndarray,
    cell_id: np.ndarray,
    *,
    inside: np.ndarray,
) -> np.ndarray:
    """Per-CAMS-cell z-score on ``frp`` (>0); one pixel in cell → 1.0; rescale to [0,1] in cell."""
    out = np.zeros(frp.shape, dtype=np.float32)
    use = inside & (frp > 0) & (cell_id >= 0)
    if not use.any():
        return out

    for cid in np.unique(cell_id[use]):
        m = use & (cell_id == int(cid))
        n = int(m.sum())
        if n == 1:
            out[m] = 1.0
            continue
        v = frp[m].astype(np.float32)
        mu = float(v.mean())
        sigma = float(v.std())
        if sigma <= 0.0:
            out[m] = 1.0
            continue
        z = (v - mu) / sigma
        z_min = float(z.min())
        z_max = float(z.max())
        if z_max <= z_min:
            out[m] = 1.0
        else:
            out[m] = ((z - z_min) / (z_max - z_min)).astype(np.float32)
    return out
