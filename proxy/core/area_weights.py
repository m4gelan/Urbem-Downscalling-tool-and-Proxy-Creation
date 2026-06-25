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
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


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
    w2: float
) -> np.ndarray:
    # S_pp = w1 * S_pop + w2 * S_u121 * S_pop, where S_u121 == corine_value_01, S_pop == pop_z
    p = np.maximum(pop_z.astype(np.float32, copy=False), 0.0)
    c = np.clip(corine_value_01.astype(np.float32, copy=False), 0.0, 1.0)
    return (w1 * p + w2 * c * p).astype(np.float32, copy=False)


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
    population_raster_residual: np.ndarray,
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
    S_residual = residual_w1 * population_raster_residual + residual_w2 * rural_mask + residual_w3 * imperviousness_raster

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
    pop = np.nan_to_num(np.asarray(population_z, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
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
    corine_132: np.ndarray,
    corine_133: np.ndarray,
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
    c132 = _clip_01(corine_132)
    c133 = _clip_01(corine_133)

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

    wm = _strip_weight_row(weights.get("manufacturing_mobile"))
    S_osm_mfg = _clip_01(osm_rasters_by_subgroup["manufacturing_mobile"]["industrial_sites"])
    S_cor_mfg = c121 + c131 + c132 + c133
    out["manufacturing_mobile"] = (
        float(wm["w_osm"]) * S_osm_mfg + float(wm["w_corine"]) * _clip_01(S_cor_mfg)
    ).astype(np.float32)

    wo = _strip_weight_row(weights.get("other_mobile"))
    S_osm_mil = _clip_01(osm_rasters_by_subgroup["other_mobile"]["military"])
    S_cor_port = c123 + c124
    out["other_mobile"] = (
        float(wo["w_osm"]) * S_osm_mil + float(wo["w_corine"]) * _clip_01(S_cor_port)
    ).astype(np.float32)

    return out


def compute_i_mobile_S_by_subgroup(
    repo_root: Path,
    corine_filepath: str | Path,
    *,
    mobile_cfg: dict[str, Any],
    corine_band: int,
    cams_cells: dict[int, dict[str, Any]],
    ref_height: int,
    ref_width: int,
    ref_transform: Any,
    ref_crs: Any,
    pop_z: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """GNFR I mobile (1A4*): weighted CORINE masks + population blend per ``mobile_masks`` in sector YAML."""
    from proxy.dataset_loaders.load_corine import load_corine_weighted_l3

    f32 = np.float32
    cor_path = repo_root / str(corine_filepath).replace("\\", "/")
    grid_kw = dict(
        ref_height=ref_height,
        ref_width=ref_width,
        ref_transform=ref_transform,
        ref_crs=ref_crs,
    )
    p = np.maximum(np.asarray(pop_z, dtype=np.float32), 0.0)
    out: dict[str, np.ndarray] = {}
    mix: dict[str, dict[str, Any]] = {}

    ag_cfg = mobile_cfg.get("agriculture_forestry_mobile")
    if not isinstance(ag_cfg, dict):
        raise ValueError("mobile_masks.agriculture_forestry_mobile required")
    S_agri = load_corine_weighted_l3(
        cor_path,
        {int(k): v for k, v in _strip_weight_row(ag_cfg["agri_l3_weights"]).items()},
        corine_band,
        cams_cells,
        **grid_kw,
    )
    S_forest = load_corine_weighted_l3(
        cor_path,
        {int(k): v for k, v in _strip_weight_row(ag_cfg["forest_l3_weights"]).items()},
        corine_band,
        cams_cells,
        **grid_kw,
    )
    w_agri = float(ag_cfg["w_agri"])
    w_forest = float(ag_cfg["w_forest"])
    agri = _clip_01(S_agri)
    forest = _clip_01(S_forest)
    out["agriculture_forestry_mobile"] = (w_agri * agri + w_forest * forest).astype(f32)
    mix["agriculture_forestry_mobile"] = {
        "mixer": "linear",
        "weight_keys": ["w_agri", "w_forest"],
        "weights": {"w_agri": w_agri, "w_forest": w_forest},
        "terms": {"agri": agri, "forest": forest},
    }
    log.info(
        "I_Offroad agriculture_forestry_mobile S_sum=%.6g w_agri=%s w_forest=%s",
        float(out["agriculture_forestry_mobile"].sum()),
        w_agri,
        w_forest,
    )

    res_cfg = mobile_cfg.get("residential_mobile")
    if not isinstance(res_cfg, dict):
        raise ValueError("mobile_masks.residential_mobile required")
    S_cor_res = load_corine_weighted_l3(
        cor_path,
        {int(k): v for k, v in _strip_weight_row(res_cfg["corine_l3_weights"]).items()},
        corine_band,
        cams_cells,
        **grid_kw,
    )
    w_pop = float(res_cfg["w_pop"])
    w_blend = float(res_cfg["w_blend"])
    c_res = _clip_01(S_cor_res)
    out["residential_mobile"] = (w_pop * p + w_blend * c_res * p).astype(f32)
    mix["residential_mobile"] = {
        "mixer": "linear",
        "weight_keys": ["w_pop", "w_blend"],
        "weights": {"w_pop": w_pop, "w_blend": w_blend},
        "terms": {"pop": p, "corine_pop": (c_res * p).astype(f32)},
    }
    log.info(
        "I_Offroad residential_mobile S_sum=%.6g w_pop=%s w_blend=%s",
        float(out["residential_mobile"].sum()),
        w_pop,
        w_blend,
    )

    com_cfg = mobile_cfg.get("commercial_mobile")
    if not isinstance(com_cfg, dict):
        raise ValueError("mobile_masks.commercial_mobile required")
    S_cor_com = load_corine_weighted_l3(
        cor_path,
        {int(k): v for k, v in _strip_weight_row(com_cfg["corine_l3_weights"]).items()},
        corine_band,
        cams_cells,
        **grid_kw,
    )
    w_cor = float(com_cfg["w_corine"])
    w_cor_pop = float(com_cfg["w_corine_pop"])
    c_com = _clip_01(S_cor_com)
    out["commercial_mobile"] = (w_cor * c_com + w_cor_pop * c_com * p).astype(f32)
    mix["commercial_mobile"] = {
        "mixer": "linear",
        "weight_keys": ["w_corine", "w_corine_pop"],
        "weights": {"w_corine": w_cor, "w_corine_pop": w_cor_pop},
        "terms": {"corine": c_com, "corine_pop": (c_com * p).astype(f32)},
    }
    log.info(
        "I_Offroad commercial_mobile S_sum=%.6g w_corine=%s w_corine_pop=%s",
        float(out["commercial_mobile"].sum()),
        w_cor,
        w_cor_pop,
    )
    return out, mix


def build_i_offroad_mix_by_group(
    *,
    osm_rasters_by_subgroup: dict[str, dict[str, np.ndarray]],
    weights_cfg: dict[str, Any],
    mobile_mix: dict[str, dict[str, Any]],
    corine_121: np.ndarray,
    corine_123: np.ndarray,
    corine_124: np.ndarray,
    corine_131: np.ndarray,
    corine_132: np.ndarray,
    corine_133: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Mix terms for W_groups export / prong A(w) and B(w); must match ``compute_i_*_S_by_subgroup``."""
    f32 = np.float32
    osg = osm_rasters_by_subgroup

    wr = _strip_weight_row(weights_cfg.get("rail_transport"))
    wp = _strip_weight_row(weights_cfg.get("pipeline_transport"))
    wm = _strip_weight_row(weights_cfg.get("non_road_machinery"))
    wmf = _strip_weight_row(weights_cfg.get("manufacturing_mobile"))
    wom = _strip_weight_row(weights_cfg.get("other_mobile"))

    c121 = _clip_01(corine_121)
    c123 = _clip_01(corine_123)
    c124 = _clip_01(corine_124)
    c131 = _clip_01(corine_131)
    c132 = _clip_01(corine_132)
    c133 = _clip_01(corine_133)

    mix: dict[str, dict[str, Any]] = {
        "rail_transport": {
            "mixer": "linear",
            "weight_keys": ["w_osm_diesel", "w_osm_electric", "w_osm_yard"],
            "weights": {k: float(wr[k]) for k in ("w_osm_diesel", "w_osm_electric", "w_osm_yard")},
            "terms": {
                "rail_diesel": np.asarray(osg["rail_transport"]["rail_diesel"], dtype=f32),
                "rail_electric": np.asarray(osg["rail_transport"]["rail_electric"], dtype=f32),
                "rail_yards": np.asarray(osg["rail_transport"]["rail_yards"], dtype=f32),
            },
        },
        "pipeline_transport": {
            "mixer": "linear",
            "weight_keys": ["w_osm_oil_gas_facilities", "w_osm_pipeline_hydrocarbon"],
            "weights": {
                k: float(wp[k]) for k in ("w_osm_oil_gas_facilities", "w_osm_pipeline_hydrocarbon")
            },
            "terms": {
                "oil_gas_facilities": np.asarray(osg["pipeline_transport"]["oil_gas_facilities"], dtype=f32),
                "pipeline_hydrocarbon": np.asarray(osg["pipeline_transport"]["pipeline_hydrocarbon"], dtype=f32),
            },
        },
        "non_road_machinery": {
            "mixer": "linear",
            "weight_keys": ["w_clc_121", "w_clc_123", "w_clc_124", "w_clc_131"],
            "weights": {k: float(wm[k]) for k in ("w_clc_121", "w_clc_123", "w_clc_124", "w_clc_131")},
            "terms": {
                "clc_121": c121,
                "clc_123": c123,
                "clc_124": c124,
                "clc_131": c131,
            },
        },
        "manufacturing_mobile": {
            "mixer": "linear",
            "weight_keys": ["w_osm", "w_corine"],
            "weights": {"w_osm": float(wmf["w_osm"]), "w_corine": float(wmf["w_corine"])},
            "terms": {
                "osm_industrial": _clip_01(osg["manufacturing_mobile"]["industrial_sites"]),
                "corine_industrial": _clip_01(c121 + c131 + c132 + c133),
            },
        },
        "other_mobile": {
            "mixer": "linear",
            "weight_keys": ["w_osm", "w_corine"],
            "weights": {"w_osm": float(wom["w_osm"]), "w_corine": float(wom["w_corine"])},
            "terms": {
                "osm_military": _clip_01(osg["other_mobile"]["military"]),
                "corine_port_air": _clip_01(c123 + c124),
            },
        },
    }
    mix.update(mobile_mix)
    return mix


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
    If ``sum_{k in C} S_k == 0`` but the cell has valid pixels, ``W_j = 1 / |C|``.
    Pixels with ``cell_id < 0`` get ``0``.
    """
    if not cams_cells:
        return np.zeros(S.shape, dtype=np.float32)

    flat_s = np.nan_to_num(np.asarray(S, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).ravel()
    flat_id = np.asarray(cell_id, dtype=np.int32).ravel()
    max_k = int(max(cams_cells))
    sums = np.zeros(max_k + 1, dtype=np.float64)
    counts = np.zeros(max_k + 1, dtype=np.int64)
    n = flat_s.size

    for start in range(0, n, _CHUNK_PIXELS):
        end = min(start + _CHUNK_PIXELS, n)
        fid = flat_id[start:end]
        w = flat_s[start:end]
        valid = fid >= 0
        w0 = np.where(valid, w, np.float32(0.0))
        idx = np.maximum(fid, 0)
        np.add.at(sums, idx, w0)
        np.add.at(counts, idx, valid.astype(np.int64))

    out_flat = np.zeros(n, dtype=np.float32)
    for start in range(0, n, _CHUNK_PIXELS):
        end = min(start + _CHUNK_PIXELS, n)
        fid = flat_id[start:end]
        w = flat_s[start:end]
        valid = fid >= 0
        cid = np.maximum(fid, 0)
        d = sums[cid]
        c = counts[cid]
        pos = valid & (d > 0.0)
        np.divide(w, d, out=out_flat[start:end], where=pos)
        uniform = valid & (d <= 0.0) & (c > 0)
        out_flat[start:end][uniform] = (1.0 / c[uniform]).astype(np.float32)

    return out_flat.reshape(S.shape)


def fuse_alpha_weighted_W_planes(
    W_planes: list[np.ndarray],
    alpha_row: np.ndarray,
    cell_id: np.ndarray,
    cams_cells: dict[int, dict[str, Any]],
) -> np.ndarray:
    """``sum_i alpha[i] * W_planes[i]`` then per-CAMS-cell normalization (export sums to 1 per cell)."""
    if not W_planes:
        raise ValueError("W_planes must be non-empty")
    alphas = np.asarray(alpha_row, dtype=np.float32).ravel()
    if alphas.shape[0] != len(W_planes):
        raise ValueError(f"alpha length {alphas.shape[0]} != {len(W_planes)} weight planes")
    w0 = np.asarray(W_planes[0], dtype=np.float32)
    acc = np.empty_like(w0)
    scratch = np.empty_like(w0)
    np.multiply(w0, float(alphas[0]), out=acc)
    for a, w in zip(alphas[1:], W_planes[1:]):
        if float(a) == 0.0:
            continue
        np.multiply(np.asarray(w, dtype=np.float32), float(a), out=scratch)
        np.add(acc, scratch, out=acc)
    return normalize_W_per_cams_cell(acc, cell_id, cams_cells)
