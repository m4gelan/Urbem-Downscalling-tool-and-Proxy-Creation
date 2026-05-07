from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from .constants import MODEL_CLASSES


def warp_band_to_ref(
    src_path: Path,
    ref: dict[str, Any],
    *,
    resampling: Resampling,
) -> np.ndarray:
    h, w = int(ref["height"]), int(ref["width"])
    dst = np.zeros((h, w), dtype=np.float32)
    with rasterio.open(src_path) as src:
        nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref["transform"],
            dst_crs=ref["crs"],
            resampling=resampling,
            src_nodata=nodata,
            dst_nodata=np.nan,
        )
    out = np.nan_to_num(dst, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.maximum(out, 0.0)


def combine_base(
    heat: np.ndarray,
    gfa: np.ndarray,
    *,
    heat_exp: float,
    gfa_exp: float,
    use_additive: bool,
    add_w_heat: float,
    add_w_gfa: float,
    epsilon: float,
) -> np.ndarray:
    H = np.maximum(heat.astype(np.float64), 0.0)
    G = np.maximum(gfa.astype(np.float64), 0.0)
    if use_additive:
        return (add_w_heat * H + add_w_gfa * G).astype(np.float32)
    e = max(float(epsilon), 1e-30)
    R = np.power(H + e, float(heat_exp)) * np.power(G + e, float(gfa_exp))
    if not np.all(np.isfinite(R)):
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    return R.astype(np.float32)


def build_clc_indicators(clc: np.ndarray, c111: int, c112: int, c121: int) -> tuple[np.ndarray, ...]:
    z = np.rint(clc).astype(np.int32)
    u111 = (z == int(c111)).astype(np.float32)
    u112 = (z == int(c112)).astype(np.float32)
    u121 = (z == int(c121)).astype(np.float32)
    return u111, u112, u121


def morph_residential(u111: np.ndarray, u112: np.ndarray, u121: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    one = np.ones_like(u111, dtype=np.float32)
    other = one - u111 - u112 - u121
    other = np.clip(other, 0.0, 1.0)
    return (
        float(cfg["w111"]) * u111
        + float(cfg["w112"]) * u112
        + float(cfg["w_other"]) * other
    ).astype(np.float32)


def morph_commercial(u111: np.ndarray, u112: np.ndarray, u121: np.ndarray, cfg: dict[str, float]) -> np.ndarray:
    one = np.ones_like(u111, dtype=np.float32)
    other = one - u111 - u112 - u121
    other = np.clip(other, 0.0, 1.0)
    return (
        float(cfg["w111"]) * u111
        + float(cfg["w121"]) * u121
        + float(cfg["w_other"]) * other
    ).astype(np.float32)


def build_X_stack(
    R_base: np.ndarray,
    C_base: np.ndarray,
    u111: np.ndarray,
    u112: np.ndarray,
    u121: np.ndarray,
    morph_cfg: dict[str, Any],
) -> np.ndarray:
    """
    X[..., k] aligned with MODEL_CLASSES order. Shape (H, W, K).
    """
    mr = morph_residential(u111, u112, u121, morph_cfg["residential_fireplace_heating_stove"])
    mc = morph_commercial(u111, u112, u121, morph_cfg["commercial_boilers"])
    h, w = R_base.shape
    k = len(MODEL_CLASSES)
    X = np.zeros((h, w, k), dtype=np.float32)
    R = R_base.astype(np.float32)
    C = C_base.astype(np.float32)
    X[:, :, 0] = R * mr
    X[:, :, 1] = R * mr
    X[:, :, 2] = R
    X[:, :, 3] = R
    X[:, :, 4] = R
    X[:, :, 5] = C * mc
    X[:, :, 6] = C * mc
    return X


def load_and_build_fields(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
) -> dict[str, Any]:
    paths = cfg["paths"]
    hm = paths["hotmaps"]
    p_res = root / hm["heat_res"]
    p_nres = root / hm["heat_nonres"]
    p_gfa_r = root / hm["gfa_res"]
    p_gfa_c = root / hm["gfa_nonres"]
    for label, p in (
        ("heat_res", p_res),
        ("heat_nonres", p_nres),
        ("gfa_res", p_gfa_r),
        ("gfa_nonres", p_gfa_c),
    ):
        if not p.is_file():
            raise FileNotFoundError(
                f"Hotmaps raster missing ({label}): {p}. Run "
                "Residential/auxiliaries/download_hotmaps_building_rasters.py"
            )

    H_res = warp_band_to_ref(p_res, ref, resampling=Resampling.bilinear)
    H_com = warp_band_to_ref(p_nres, ref, resampling=Resampling.bilinear)
    G_res = warp_band_to_ref(p_gfa_r, ref, resampling=Resampling.bilinear)
    G_com = warp_band_to_ref(p_gfa_c, ref, resampling=Resampling.bilinear)

    bp = cfg["base_proxy"]
    R_base = combine_base(
        H_res,
        G_res,
        heat_exp=float(bp["heat_exp"]),
        gfa_exp=float(bp["gfa_exp"]),
        use_additive=bool(bp["use_additive_instead"]),
        add_w_heat=float(bp["additive_heat_weight"]),
        add_w_gfa=float(bp["additive_gfa_weight"]),
        epsilon=float(bp["epsilon"]),
    )
    C_base = combine_base(
        H_com,
        G_com,
        heat_exp=float(bp["heat_exp"]),
        gfa_exp=float(bp["gfa_exp"]),
        use_additive=bool(bp["use_additive_instead"]),
        add_w_heat=float(bp["additive_heat_weight"]),
        add_w_gfa=float(bp["additive_gfa_weight"]),
        epsilon=float(bp["epsilon"]),
    )

    corine_path = Path(ref["corine_path"])
    clc = warp_band_to_ref(corine_path, ref, resampling=Resampling.nearest)
    mc = cfg["morphology"]
    u111, u112, u121 = build_clc_indicators(
        clc,
        int(mc["urban_111"]),
        int(mc["urban_112"]),
        int(mc["urban_121"]),
    )
    X = build_X_stack(R_base, C_base, u111, u112, u121, mc)
    return {
        "R_base": R_base,
        "C_base": C_base,
        "clc": clc,
        "X": X,
    }
