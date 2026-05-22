from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from proxy.alpha.Compute_alpha_matrix import AlphaMatrixResult
from proxy.core import log
from proxy.core.raster_helpers import warp_raster_to_grid
from proxy.dataset_loaders.load_corine import load_corine

_OFFROAD_EPS = 1e-15

def _strip_weight_row(row: Any) -> dict[str, float]:
    return {str(k).strip(): float(v) for k, v in (row or {}).items()}


def _clip_01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def _osm_term(
    osm_by_subgroup: dict[str, dict[str, np.ndarray]],
    subgroup: str,
    slot_weight_keys: list[tuple[str, str]],
    w: dict[str, float],
) -> np.ndarray:
    """``sum_i w[key_i] * OSM_slot_i`` (slots must exist under ``osm_by_subgroup[subgroup]``)."""
    slots = osm_by_subgroup[subgroup]
    acc = None
    for slot_id, wkey in slot_weight_keys:
        if slot_id not in slots:
            raise KeyError(
                f"OSM missing slot {slot_id!r} under subgroup {subgroup!r} "
                f"(have {sorted(slots.keys())})"
            )
        term = float(w[wkey]) * _clip_01(slots[slot_id])
        acc = term if acc is None else acc + term
    if acc is None:
        raise ValueError(f"no OSM slot weights for subgroup {subgroup!r}")
    return acc


def combined_S_publicpower(
    pop_z: np.ndarray,
    corine_value_01: np.ndarray,
    *,
    w1: float,
    w2: float,
    delta: float,
) -> np.ndarray:
    """
    ``S = w1 * pop_z + w2 * pop_z ** (1 - delta * CORINE_value)``.

    ``CORINE_value`` is treated as eligibility in ``{0, 1}`` (from the sector CORINE mask).
    """
    p = np.maximum(pop_z.astype(np.float32, copy=False), np.float32(0.0))
    c = np.clip(corine_value_01.astype(np.float32, copy=False), np.float32(0.0), np.float32(1.0))
    exp = np.maximum(np.float32(1.0) - np.float32(delta) * c, np.float32(1e-12))
    term2 = np.power(p, exp)
    return (np.float32(w1) * p + np.float32(w2) * term2).astype(np.float32, copy=False)


def combined_S_shipping(
    corine_01: np.ndarray,
    emodnet_z: np.ndarray,
    osm_raster: np.ndarray,
    *,
    w1: float,
    w2: float,
    w3: float,
) -> np.ndarray:
    """``S = w1 * emodnet_z + w2 * osm + w3 * corine`` (all clipped to sensible ranges)."""
    c = np.clip(corine_01.astype(np.float64), 0.0, 1.0)
    e = np.maximum(emodnet_z.astype(np.float64), 0.0)
    o = np.maximum(osm_raster.astype(np.float64), 0.0)
    return w1 * e + w2 * o + w3 * c


def combined_S_waste(

    corine_map_w1: np.ndarray,
    corine_map_w2: np.ndarray,
    osm_raster: np.ndarray,
    uwwtd_agg_raster: np.ndarray,
    uwwtd_plants_raster: np.ndarray,
    imperviousness_raster: np.ndarray,
    rural_mask: np.ndarray,
    population_raster: np.ndarray,
    solid_waste_w1: float   ,
    solid_waste_w2: float,
    solid_waste_w3: float,
    wastewater_w1: float,
    wastewater_w2: float,
    wastewater_w3: float,
    wastewater_w4: float,
    residual_w1: float,
    residual_w2: float,
    residual_w3: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the 3 S for waste sectors
    """
    S_solid_waste = solid_waste_w1 * corine_map_w1 + solid_waste_w2 * corine_map_w2 + solid_waste_w3 * osm_raster
    S_wastewater = wastewater_w1 * uwwtd_agg_raster + wastewater_w2 * uwwtd_plants_raster + wastewater_w3 * population_raster + wastewater_w4 * imperviousness_raster
    S_residual = residual_w1 * population_raster + residual_w2 * rural_mask + residual_w3 * imperviousness_raster

    return S_solid_waste, S_wastewater, S_residual

def compute_E_solvents_S_by_activity(
    *,
    osm_rasters_by_subgroup: dict[str, dict[str, np.ndarray]],
    corine_map_household: np.ndarray,
    corine_map_service: np.ndarray,
    corine_map_industrial: np.ndarray,
    corine_map_transport: np.ndarray,
    population_z: np.ndarray,
    weights_cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Linear mixture ``S`` per activity archetype (household, service, industrial, infrastructure).
    """
    pop = np.nan_to_num(np.asarray(population_z, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    c_household = _clip_01(corine_map_household)
    c_service = _clip_01(corine_map_service)
    c_industrial = _clip_01(corine_map_industrial)
    c_transport = _clip_01(corine_map_transport)

    out: dict[str, np.ndarray] = {}

    w1 = _strip_weight_row(weights_cfg.get("household"))
    out["household"] = float(w1["w_corine"]) * c_household + float(w1["w_population"]) * pop

    w2 = _strip_weight_row(weights_cfg.get("service"))
    out["service"] = (
        float(w2["w_corine_urban"]) * c_household
        + float(w2["w_corine_service"]) * c_service
        + _osm_term(osm_rasters_by_subgroup, "service", [("service", "w_osm_service")], w2)
    )

    w3 = _strip_weight_row(weights_cfg.get("industrial"))
    out["industrial"] = float(w3["w_corine_industrial"]) * c_industrial + _osm_term(
        osm_rasters_by_subgroup, "industrial", [("industrial", "w_osm_industrial")], w3
    )

    w4 = _strip_weight_row(weights_cfg.get("infrastructure"))
    out["infrastructure"] = float(w4["w_corine_transport"]) * c_transport + _osm_term(
        osm_rasters_by_subgroup, "roads", [("roads", "w_osm_roads")], w4
    )

    return out


def combine_S_solvents_subsectors(
    S_by_activity: dict[str, np.ndarray],
    beta: dict[str, Any],
    *,
    activity_keys: list[str],
    subsector_order: list[str] | tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """``S_s = sum_a beta[s,a] * S_a`` for each solvent subsector ``s``."""
    if not activity_keys:
        raise ValueError("activity_keys must be non-empty (from sector config weights)")
    order = list(subsector_order) if subsector_order else list(beta.keys())
    if not order:
        raise ValueError("beta block is empty")
    out: dict[str, np.ndarray] = {}
    for sname in order:
        row = beta.get(sname)
        if not isinstance(row, dict):
            raise ValueError(f"beta.{sname} must be a mapping of activity weights")
        wrow = _strip_weight_row(row)
        acc: np.ndarray | None = None
        for act in activity_keys:
            if act not in S_by_activity:
                raise KeyError(f"missing activity S raster {act!r}")
            if act not in wrow:
                raise KeyError(f"beta.{sname} missing activity weight {act!r}")
            term = float(wrow[act]) * S_by_activity[act]
            acc = term if acc is None else acc + term
        out[sname] = acc  # type: ignore[assignment]
    return out



def compute_d_fugitive_S_by_subgroup(
    *,
    osm_rasters_by_subgroup: dict[str, dict[str, np.ndarray]],
    corine_121: np.ndarray,
    corine_123: np.ndarray,
    corine_131: np.ndarray,
    population_z: np.ndarray,
    gem_coal: np.ndarray,
    gem_oil: np.ndarray,
    vnf: np.ndarray,
    weights: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Linear mixture ``S`` per inventory subgroup (GNFR D fugitive), using ``weights.*`` in the sector YAML.

    OSM slots are combined with weights ``w_osm_*``; binary rasters and CORINE masks are clipped to ``[0, 1]``;
    population uses ``population_z`` with NaNs cleared like industry.
    """
    pop = np.nan_to_num(np.asarray(population_z, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    c121 = _clip_01(corine_121)
    c123 = _clip_01(corine_123)
    c131 = _clip_01(corine_131)
    gcoal = _clip_01(gem_coal)
    goil = _clip_01(gem_oil)
    vn = _clip_01(vnf)

    out: dict[str, np.ndarray] = {}

    w1 = _strip_weight_row(weights.get("coal_and_solid_fuels"))
    out["coal_and_solid_fuels"] = (
        _osm_term(
            osm_rasters_by_subgroup,
            "coal_and_solid_fuels",
            [("quarry_coal_mine", "w_osm")],
            w1,
        )
        + float(w1["w_clc_131"]) * c131
        + float(w1["w_clc_121"]) * c121
        + float(w1["w_gem_coal"]) * gcoal
    )

    w2 = _strip_weight_row(weights.get("oil_upstream_and_transport"))
    out["oil_upstream_and_transport"] = (
        _osm_term(
            osm_rasters_by_subgroup,
            "oil_upstream_and_transport",
            [("pipeline_well", "w_osm_pipeline"), ("port_oil_depot", "w_osm_port")],
            w2,
        )
        + float(w2["w_clc_121"]) * c121
        + float(w2["w_clc_123"]) * c123
        + float(w2["w_gem_oil"]) * goil
    )

    w3 = _strip_weight_row(weights.get("storage_refining_distribution"))
    out["storage_refining_distribution"] = (
        _osm_term(
            osm_rasters_by_subgroup,
            "storage_refining_distribution",
            [("refinery", "w_osm_refinery"), ("tank_storage", "w_osm_tank"), ("fuel_depot", "w_osm_fuel")],
            w3,
        )
        + float(w3["w_clc_121"]) * c121
        + float(w3["w_pop"]) * pop
    )

    w4 = _strip_weight_row(weights.get("gas_flaring_and_residual_losses"))
    out["gas_flaring_and_residual_losses"] = (
        _osm_term(
            osm_rasters_by_subgroup,
            "gas_flaring_and_residual_losses",
            [("flare_chimney", "w_osm_flaring"), ("power_gen", "w_osm_power")],
            w4,
        )
        + float(w4["w_viirs"]) * vn
        + float(w4["w_clc_121"]) * c121
        + float(w4["w_clc_123"]) * c123
    )

    return out

def compute_i_offroad_S_by_subgroup(
    *,
    osm_rasters_by_subgroup: dict[str, dict[str, np.ndarray]],
    corine_121: np.ndarray,
    corine_123: np.ndarray,
    corine_131: np.ndarray,
    corine_124: np.ndarray,
    weights: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Linear mixture ``S`` per inventory subgroup (GNFR I Offroad), using ``weights.*`` in the sector YAML.

    OSM slots are combined with weights ``w_osm_*``; binary rasters and CORINE masks are clipped to ``[0, 1]``.
    """
    c121 = _clip_01(corine_121)
    c123 = _clip_01(corine_123)
    c131 = _clip_01(corine_131)
    c124 = _clip_01(corine_124)

    out: dict[str, np.ndarray] = {}

    w1 = _strip_weight_row(weights.get("rail_transport"))
    out["rail_transport"] = (
        _osm_term(
            osm_rasters_by_subgroup,
            "rail_transport",
            [("rail_diesel", "w_osm_diesel"), ("rail_electric", "w_osm_electric"), ("rail_yards", "w_osm_yard")],
            w1,
        )
    )
    w2 = _strip_weight_row(weights.get("pipeline_transport"))
    out["pipeline_transport"] = (
        _osm_term(
            osm_rasters_by_subgroup,
            "pipeline_transport",
            [("oil_gas_facilities", "w_osm_oil_gas_facilities"), ("pipeline_hydrocarbon", "w_osm_pipeline_hydrocarbon")],
            w2,
        )
    )
    w3 = _strip_weight_row(weights.get("non_road_machinery"))
    out["non_road_machinery"] = (float(w3["w_clc_121"]) * c121 + float(w3["w_clc_123"]) * c123 + float(w3["w_clc_124"]) * c124 + float(w3["w_clc_131"]) * c131)
    return out

def combined_S_industry_group(
    osm_01: np.ndarray,
    corine_01: np.ndarray,
    pop_z: np.ndarray,
    *,
    w_osm: float,
    w_clc: float,
    w_pop: float,
) -> np.ndarray:
    """
    ``S = (1 - w_pop) * (w_osm * OSM + w_clc * CORINE) + w_pop * Z_population``.

    ``OSM`` and ``CORINE`` are treated as eligibility in ``[0, 1]``; ``Z_population`` is a z-scored raster
    (NaNs treated as 0 after the caller's preprocessing).
    """
    o = np.clip(osm_01.astype(np.float64), 0.0, 1.0)
    c = np.clip(corine_01.astype(np.float64), 0.0, 1.0)
    z = np.nan_to_num(pop_z.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    w_o = float(w_osm)
    w_c = float(w_clc)
    w_p = float(w_pop)
    spatial = w_o * o + w_c * c
    return (1.0 - w_p) * spatial + w_p * z


_CHUNK_PIXELS = 4_000_000


def normalize_W_per_cams_cell(
    S: np.ndarray,
    cell_id: np.ndarray,
    cams_cells: dict[int, dict[str, Any]],
) -> np.ndarray:
    """
    ``W_j = S_j / sum_{k in C} S_k`` for pixels ``j`` in the same CAMS cell ``C``.
    Pixels with ``cell_id < 0`` get ``0``.

    ``cell_id`` is expected country-restricted (invalid ids already ``-1``); chunked to limit peak RAM.
    """
    if not cams_cells:
        return np.zeros(S.shape, dtype=np.float32)

    flat_s = np.nan_to_num(np.asarray(S, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).ravel()
    flat_id = np.asarray(cell_id, dtype=np.int32).ravel()
    max_k = int(max(cams_cells))
    sums = np.zeros(max_k + 1, dtype=np.float64)
    n = flat_s.size

    for start in range(0, n, _CHUNK_PIXELS):
        end = min(start + _CHUNK_PIXELS, n)
        fid = flat_id[start:end]
        w = flat_s[start:end]
        w0 = np.where(fid >= 0, w, np.float32(0.0))
        np.add.at(sums, np.maximum(fid, 0), w0)

    if not np.any(sums > 0):
        return np.zeros(S.shape, dtype=np.float32)

    out_flat = np.zeros(n, dtype=np.float32)
    for start in range(0, n, _CHUNK_PIXELS):
        end = min(start + _CHUNK_PIXELS, n)
        fid = flat_id[start:end]
        w = flat_s[start:end]
        d = sums[np.maximum(fid, 0)]
        np.divide(w, d, out=out_flat[start:end], where=(fid >= 0) & (d > 0.0))

    return out_flat.reshape(S.shape)


def combined_S_offroad_branches(
    S_forest: np.ndarray,
    S_residential: np.ndarray,
    S_commercial: np.ndarray,
    *,
    beta_F: float,
    beta_R: float,
    beta_B: float,
) -> np.ndarray:
    """Weighted sum of three off-road eligibility / score rasters."""
    return (
        float(beta_F) * S_forest.astype(np.float64)
        + float(beta_R) * S_residential.astype(np.float64)
        + float(beta_B) * S_commercial.astype(np.float64)
    ).astype(np.float32)


def _offroad_branch_betas(
    alpha_row: np.ndarray,
    group_index: dict[str, int],
) -> tuple[float, float, float, float, float]:
    """(alpha_stat, alpha_off, beta_F, beta_R, beta_B) from one pollutant α row."""
    a_stat = float(alpha_row[group_index["stationary"]])
    a_for = float(alpha_row[group_index["forestry_offroad"]])
    a_res = float(alpha_row[group_index["residential_offroad"]])
    a_com = float(alpha_row[group_index["commercial_offroad"]])
    a_off = a_for + a_res + a_com
    if a_off <= _OFFROAD_EPS:
        return 1.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0
    return a_stat, a_off, a_for / a_off, a_res / a_off, a_com / a_off


def fuse_offroad_combined_band(
    W_stationary: np.ndarray,
    S_forest: np.ndarray,
    S_residential: np.ndarray,
    S_commercial: np.ndarray,
    cell_id: np.ndarray,
    cams_cells: dict[int, dict[str, Any]],
    alpha_row: np.ndarray,
    group_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One pollutant: W_off (CAMS-norm) and W_combined = α_stat·W_stat + α_off·W_off."""
    a_stat, a_off, bF, bR, bB = _offroad_branch_betas(alpha_row, group_index)
    S_off = combined_S_offroad_branches(S_forest, S_residential, S_commercial, beta_F=bF, beta_R=bR, beta_B=bB)
    W_o = normalize_W_per_cams_cell(S_off, cell_id, cams_cells)
    W_s = np.asarray(W_stationary, dtype=np.float32)
    W_c = (a_stat * W_s + a_off * W_o).astype(np.float32)
    return W_s, W_o, W_c


def fuse_stationary_offroad_weights(
    W_stationary: np.ndarray,
    S_forest: np.ndarray,
    S_residential: np.ndarray,
    S_commercial: np.ndarray,
    cell_id: np.ndarray,
    cams_cells: dict[int, dict[str, Any]],
    alpha_result: AlphaMatrixResult,
    pollutant_labels: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """All pollutants at once (needs h×w×p); prefer per-band loop in C_Othercombustion pipeline."""
    gix = {g: i for i, g in enumerate(alpha_result.group_names)}
    for g in ("stationary", "forestry_offroad", "residential_offroad", "commercial_offroad"):
        if g not in gix:
            raise ValueError(f"alpha groups missing {g!r}; have {alpha_result.group_names}")

    h, w, n_p = W_stationary.shape
    W_fused = np.zeros((h, w, n_p), dtype=np.float32)
    w_stat: dict[str, np.ndarray] = {}
    w_off: dict[str, np.ndarray] = {}
    w_comb: dict[str, np.ndarray] = {}

    for j, pol in enumerate(pollutant_labels):
        W_s, W_o, W_c = fuse_offroad_combined_band(
            W_stationary[:, :, j], S_forest, S_residential, S_commercial,
            cell_id, cams_cells, alpha_result.alpha[j], gix,
        )
        W_fused[:, :, j] = W_c
        w_stat[pol] = W_s
        w_off[pol] = W_o
        w_comb[pol] = W_c

    return np.transpose(W_fused, (2, 0, 1)), w_stat, w_off, w_comb


def log_stationary_alpha(
    alpha_result: AlphaMatrixResult,
    pollutant_labels: list[str],
) -> None:
    gix = {g: i for i, g in enumerate(alpha_result.group_names)}
    log.info("--- stationary branch ---")
    for j, pol in enumerate(pollutant_labels):
        row = alpha_result.alpha[j]
        log.info(f"  {pol}: alpha={float(row[gix['stationary']]):.6g}")


def _log_branch_alpha_beta(
    alpha_result: AlphaMatrixResult,
    pollutant_labels: list[str],
    branch: str,
) -> None:
    gix = {g: i for i, g in enumerate(alpha_result.group_names)}
    group_key = {
        "forestry": "forestry_offroad",
        "residential": "residential_offroad",
        "commercial": "commercial_offroad",
    }[branch]
    for j, pol in enumerate(pollutant_labels):
        row = alpha_result.alpha[j]
        _, _, bF, bR, bB = _offroad_branch_betas(row, gix)
        beta = {"forestry": bF, "residential": bR, "commercial": bB}[branch]
        log.info(
            f"  {pol}: alpha={float(row[gix[group_key]]):.6g} beta={beta:.6g}"
        )


def build_offroad_S_from_corine(
    repo_root: Path,
    corine_filepath: str | Path,
    *,
    l3_forest: list[int],
    l3_residential: list[int],
    l3_commercial: list[int],
    corine_band: int,
    cams_cells: dict[int, dict[str, Any]],
    cams_grid: dict[str, Any],
    ref_height: int,
    ref_width: int,
    ref_transform: Any,
    ref_crs: Any,
    pop_z: np.ndarray,
    residential_w1: float,
    residential_w2: float,
    residential_delta: float,
    alpha_result: AlphaMatrixResult | None = None,
    pollutant_labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forest, residential (pop+CORINE), commercial masks on a reference grid."""
    cor_path = repo_root / str(corine_filepath).replace("\\", "/")

    def _mask(l3_codes: list[int]) -> np.ndarray:
        m, tr, crs, _ = load_corine(
            cor_path, l3_codes, corine_band, cams_cells, cams_grid, need_cell_id=False,
        )
        return warp_raster_to_grid(
            m, tr, crs, ref_height, ref_width, ref_transform, ref_crs, dest_init_nan=False,
        )

    log.info("--- building mask for forestry ---")
    S_forest = _mask(l3_forest)
    log.info(f"  S sum={float(S_forest.sum()):.6g}")
    if alpha_result is not None and pollutant_labels:
        _log_branch_alpha_beta(alpha_result, pollutant_labels, "forestry")

    log.info("--- building mask for residential ---")
    cor_res = _mask(l3_residential)
    S_residential = combined_S_publicpower(
        pop_z, cor_res, w1=residential_w1, w2=residential_w2, delta=residential_delta,
    )
    log.info(
        f"  S sum={float(S_residential.sum()):.6g} "
        f"w1={residential_w1} w2={residential_w2} delta={residential_delta}"
    )
    if alpha_result is not None and pollutant_labels:
        _log_branch_alpha_beta(alpha_result, pollutant_labels, "residential")

    log.info("--- building mask for commercial ---")
    S_commercial = _mask(l3_commercial)
    log.info(f"  S sum={float(S_commercial.sum()):.6g}")
    if alpha_result is not None and pollutant_labels:
        _log_branch_alpha_beta(alpha_result, pollutant_labels, "commercial")
    return S_forest, S_residential, S_commercial
