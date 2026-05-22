from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.enums import Resampling

from proxy.core import log
from proxy.core.raster_helpers import warp_raster_to_grid
from proxy.core.z_score import z_score_inside
from proxy.dataset_loaders.load_cams_cells_mask import load_cams_cells_mask, pixels_inside_cams_cells
from proxy.dataset_loaders.load_corine import load_corine
from proxy.dataset_loaders.load_hotmaps import (
    load_hotmaps_hdd,
    load_hotmaps_heat_nres,
    load_hotmaps_heat_res,
)
from proxy.dataset_loaders.load_population import load_population
from proxy.dataset_loaders.load_waste_rasters import load_ghsl_smod
from proxy.sector.C_Othercombustion.M_matrix import MODEL_CLASSES

@dataclass(frozen=True)
class XBuildResult:
    X: np.ndarray
    transform: Any
    crs: Any
    cell_id: np.ndarray
    cams_cells: dict[int, dict[str, Any]]
    cams_grid: dict[str, Any]
    pop_z: np.ndarray
    H_res_z: np.ndarray
    H_nres_z: np.ndarray
    Hdd_z: np.ndarray
    u111: np.ndarray
    u112: np.ndarray
    u121: np.ndarray
    stock_by_class: dict[str, np.ndarray]
    load_by_class: dict[str, np.ndarray]


_CLASS_STOCK_KEY: dict[str, tuple[str, str]] = {
    "R_FIREPLACE": ("residential", "fireplace"),
    "R_HEATING_STOVE": ("residential", "heating_stove"),
    "R_COOKING_STOVE": ("residential", "cooking_stove"),
    "R_BOILER_MAN": ("residential", "boiler_man"),
    "R_BOILER_AUT": ("residential", "boiler_aut"),
    "C_BOILER_MAN": ("commercial", "boiler_man"),
    "C_BOILER_AUT": ("commercial", "boiler_aut"),
}


def _u_rural(
    ghsl_rural: np.ndarray,
    u111: np.ndarray,
    u112: np.ndarray,
    u121: np.ndarray,
) -> np.ndarray:
    not_urb = (u111 < 0.5) & (u112 < 0.5) & (u121 < 0.5)
    return ((ghsl_rural > 0.5) & not_urb).astype(np.float32)


def _u_other(u111: np.ndarray, u112: np.ndarray, u121: np.ndarray) -> np.ndarray:
    u = np.asarray(u111, dtype=np.float32)
    return np.clip(1.0 - u - np.asarray(u112, dtype=np.float32) - np.asarray(u121, dtype=np.float32), 0.0, 1.0)


def stock_support(
    class_name: str,
    stock_cfg: dict[str, Any],
    *,
    pop_z: np.ndarray,
    u111: np.ndarray,
    u112: np.ndarray,
    u121: np.ndarray,
    u_rural: np.ndarray,
    u_other: np.ndarray,
) -> np.ndarray:
    sector, key = _CLASS_STOCK_KEY[class_name]
    w = stock_cfg[sector][key]
    w111 = float(w["weight_corine_111"])
    w112 = float(w["weight_corine_112"])
    wr = float(w["weight_ghsl_smod"])
    acc = w111 * u111 + w112 * u112 + wr * u_rural
    if sector == "commercial":
        w121 = float(w["weight_corine_121"])
        acc = acc + w121 * u121 + wr * u_other
        return acc.astype(np.float32)
    return (pop_z * acc).astype(np.float32)


def load_support(
    class_name: str,
    load_cfg: dict[str, Any],
    *,
    H_res_z: np.ndarray,
    H_nres_z: np.ndarray,
    Hdd_z: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    e_heat = float(load_cfg["exponent_heat"])
    e_hdd = float(load_cfg["exponent_hdd"])
    if class_name in ("R_FIREPLACE", "R_HEATING_STOVE"):
        return (np.power(H_res_z, e_heat) * np.power(Hdd_z, e_hdd)).astype(np.float32)
    if class_name == "R_COOKING_STOVE":
        return np.ones(shape, dtype=np.float32)
    if class_name in ("R_BOILER_MAN", "R_BOILER_AUT"):
        return H_res_z.astype(np.float32, copy=False)
    if class_name in ("C_BOILER_MAN", "C_BOILER_AUT"):
        return H_nres_z.astype(np.float32, copy=False)
    raise ValueError(f"unknown class for load_support: {class_name!r}")


def build_x_matrix(
    repo_root: Path,
    cfg: dict[str, Any],
    country_profile: dict[str, str],
    *,
    crs: str,
    resolution_m: float,
    pad_m: float,
    cams_filepath: str | Path,
    corine_filepath: str | Path,
    population_filepath: str | Path,
    ghs_smod_filepath: str | Path,
    hotmaps_residential_filepath: str | Path,
    hotmaps_non_residential_filepath: str | Path,
    hdd_filepath: str | Path,
    year: int,
    emission_category_indices: list[int],
    source_type_indices: list[int],
    pollutants: list[str],
) -> XBuildResult:
    """Build X (H, W, 7) with X[:,:,k] = S_k * L_k on the CORINE reference grid."""
    corine_cfg = cfg.get("corine") or {}
    ghsl_cfg = cfg.get("ghsl_smod") or {}
    stock_cfg = cfg.get("stock_support") or {}
    load_cfg = cfg.get("load_support") or {}
    corine_band = int(corine_cfg.get("band", 1))
    rural_codes = [int(x) for x in ghsl_cfg.get("rural_codes", [])]
    ghsl_band = int(ghsl_cfg.get("band", 1))
    l3_111 = [int(x) for x in corine_cfg.get("l3_code_111", [111])]
    l3_112 = [int(x) for x in corine_cfg.get("l3_code_112", [112])]
    l3_121 = [int(x) for x in corine_cfg.get("l3_code_121", [121])]

    cams_cells, cams_grid = load_cams_cells_mask(
        repo_root / str(cams_filepath).replace("\\", "/"),
        year=year,
        country_iso3=country_profile["ISO3"],
        emission_category_indices=emission_category_indices,
        source_type_indices=source_type_indices,
        pollutants=[str(p).strip() for p in pollutants if str(p).strip()],
        crs=crs,
        resolution_m=resolution_m,
        pad_m=pad_m,
    )
    if not cams_cells:
        raise ValueError("C_OtherCombustion X: no CAMS cells for country")

    u111, cor_tr, cor_crs, _ = load_corine(
        repo_root / str(corine_filepath).replace("\\", "/"),
        l3_111,
        corine_band,
        cams_cells,
        cams_grid,
        need_cell_id=False,
    )
    u112, _, _, _ = load_corine(
        repo_root / str(corine_filepath).replace("\\", "/"),
        l3_112,
        corine_band,
        cams_cells,
        cams_grid,
        need_cell_id=False,
    )
    u121, _, _, cell_id = load_corine(
        repo_root / str(corine_filepath).replace("\\", "/"),
        l3_121,
        corine_band,
        cams_cells,
        cams_grid,
    )
    ch, cw = u111.shape

    pop, _pop_inside, pop_tr, pop_crs, pop_nd = load_population(
        repo_root / str(population_filepath).replace("\\", "/"),
        cams_cells,
    )
    pop = warp_raster_to_grid(
        pop, pop_tr, pop_crs, ch, cw, cor_tr, cor_crs,
        src_nodata=pop_nd, dest_init_nan=True, nan_fill=0.0,
    )

    H_res, hres_tr, hres_crs, hres_nd = load_hotmaps_heat_res(
        repo_root / str(hotmaps_residential_filepath).replace("\\", "/"),
        cams_cells,
    )
    H_nres, hnres_tr, hnres_crs, hnres_nd = load_hotmaps_heat_nres(
        repo_root / str(hotmaps_non_residential_filepath).replace("\\", "/"),
        cams_cells,
    )
    Hdd, hdd_tr, hdd_crs, hdd_nd = load_hotmaps_hdd(
        repo_root / str(hdd_filepath).replace("\\", "/"),
        cams_cells,
    )
    ghsl_rural, ghsl_tr, ghsl_crs = load_ghsl_smod(
        repo_root / str(ghs_smod_filepath).replace("\\", "/"),
        cams_cells,
        rural_codes=rural_codes,
        band=ghsl_band,
    )
    ghsl_rural = warp_raster_to_grid(
        ghsl_rural, ghsl_tr, ghsl_crs, ch, cw, cor_tr, cor_crs,
        resampling=Resampling.nearest, dest_init_nan=False,
    )

    H_res = warp_raster_to_grid(
        H_res, hres_tr, hres_crs, ch, cw, cor_tr, cor_crs, src_nodata=hres_nd,
    )
    H_nres = warp_raster_to_grid(
        H_nres, hnres_tr, hnres_crs, ch, cw, cor_tr, cor_crs, src_nodata=hnres_nd,
    )
    Hdd = warp_raster_to_grid(
        Hdd, hdd_tr, hdd_crs, ch, cw, cor_tr, cor_crs, src_nodata=hdd_nd,
    )

    inside = pixels_inside_cams_cells(ch, cw, cor_tr, cor_crs, cams_cells)
    pop_z = z_score_inside(np.asarray(pop, dtype=np.float32), inside, upper_quantile=0.99, rescale_to_01=True)
    del pop
    H_res_z = z_score_inside(np.asarray(H_res, dtype=np.float32), inside, upper_quantile=0.99, rescale_to_01=True)
    del H_res
    H_nres_z = z_score_inside(np.asarray(H_nres, dtype=np.float32), inside, upper_quantile=0.99, rescale_to_01=True)
    del H_nres
    Hdd_z = z_score_inside(np.asarray(Hdd, dtype=np.float32), inside, upper_quantile=0.99, rescale_to_01=True)
    del Hdd

    u_rural = _u_rural(ghsl_rural, u111, u112, u121)
    u_other = _u_other(u111, u112, u121)

    X = np.zeros((ch, cw, len(MODEL_CLASSES)), dtype=np.float32)
    stock_by_class: dict[str, np.ndarray] = {}
    load_by_class: dict[str, np.ndarray] = {}
    for ki, cls in enumerate(MODEL_CLASSES):
        S = stock_support(
            cls, stock_cfg, pop_z=pop_z, u111=u111, u112=u112, u121=u121, u_rural=u_rural, u_other=u_other
        )
        L = load_support(cls, load_cfg, H_res_z=H_res_z, H_nres_z=H_nres_z, Hdd_z=Hdd_z, shape=(ch, cw))
        stock_by_class[cls] = S
        load_by_class[cls] = L
        X[:, :, ki] = (S * L).astype(np.float32)

    log.info(f"C_OtherCombustion X stack shape=({ch}, {cw}, {len(MODEL_CLASSES)}) bands={list(MODEL_CLASSES)}")
    log.info("C_OtherCombustion X matrix finished.")
    return XBuildResult(
        X=X,
        transform=cor_tr,
        crs=cor_crs,
        cell_id=cell_id,
        cams_cells=cams_cells,
        cams_grid=cams_grid,
        pop_z=pop_z,
        H_res_z=H_res_z,
        H_nres_z=H_nres_z,
        Hdd_z=Hdd_z,
        u111=u111,
        u112=u112,
        u121=u121,
        stock_by_class=stock_by_class,
        load_by_class=load_by_class,
    )
