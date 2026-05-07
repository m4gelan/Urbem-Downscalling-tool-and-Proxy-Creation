"""
Build **X** (H×W×7) from Hotmaps bases, CORINE morphology, optional rural bias.

**Role**: single entry ``load_and_build_fields`` used by the sector pipeline and
visualisation previews.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rasterio.warp import Resampling

from PROXY.core.corine.encoding import decode_corine_to_l3_pixels, normalized_corine_pixel_encoding
from PROXY.core.dataloaders import resolve_path, warp_band_to_ref

from ..constants import MODEL_CLASSES
from .._log import LOG
from .corine_morphology import morph_commercial, morph_residential, morphology_masks_from_clc
from .hotmaps_base import combine_base
from .rural_bias import rural_bias_from_density


def build_X_stack(
    R_base: np.ndarray,
    C_base: np.ndarray,
    u111: np.ndarray,
    u112: np.ndarray,
    u121: np.ndarray,
    morph_cfg: dict[str, Any],
    *,
    rural_bias: np.ndarray | None = None,
    rural_classes: tuple[int, ...] | None = None,
) -> np.ndarray:
    mr = morph_residential(u111, u112, u121, morph_cfg["residential_fireplace_heating_stove"])
    mc = morph_commercial(u111, u112, u121, morph_cfg["commercial_boilers"])
    h, w = R_base.shape
    k = len(MODEL_CLASSES)
    X = np.zeros((h, w, k), dtype=np.float32)
    R = R_base.astype(np.float32)
    C = C_base.astype(np.float32)
    rb = rural_bias
    if rb is None:
        rb_arr = np.ones((h, w), dtype=np.float32)
    else:
        rb_arr = np.asarray(rb, dtype=np.float32).reshape(h, w)
    rc = rural_classes if rural_classes is not None else (0, 1)
    for band in range(k):
        if band in rc and rb is not None:
            fac = rb_arr
        else:
            fac = np.ones((h, w), dtype=np.float32)
        if band in (0, 1):
            X[:, :, band] = R * mr * fac
        elif band in (2, 3, 4):
            X[:, :, band] = R
        else:
            X[:, :, band] = C * mc
    return X


def load_and_build_fields(
    root: Path,
    cfg: dict[str, Any],
    ref: dict[str, Any],
) -> dict[str, Any]:
    paths = cfg["paths"]
    hm = paths["hotmaps"]
    p_res = resolve_path(root, hm["heat_res"])
    p_nres = resolve_path(root, hm["heat_nonres"])
    p_gfa_r = resolve_path(root, hm["gfa_res"])
    p_gfa_c = resolve_path(root, hm["gfa_nonres"])
    for label, p in (
        ("heat_res", p_res),
        ("heat_nonres", p_nres),
        ("gfa_res", p_gfa_r),
        ("gfa_nonres", p_gfa_c),
    ):
        if not p.is_file():
            raise FileNotFoundError(
                f"Hotmaps raster missing ({label}): {p}. "
                "Check paths.yaml proxy_specific.other_combustion.hotmaps_dir and sector hotmaps filenames."
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
    cor_cfg = cfg.get("corine") or {}
    cor_band = int(cor_cfg.get("band", 1))
    clc = warp_band_to_ref(corine_path, ref, resampling=Resampling.nearest, band=cor_band)
    enc = normalized_corine_pixel_encoding(cor_cfg.get("pixel_encoding"))
    pmap = cor_cfg.get("pixel_value_map")
    clc_l3 = decode_corine_to_l3_pixels(
        clc,
        enc,
        repo_root=root,
        pixel_value_map=pmap,
    )
    mc = cfg["morphology"]
    u111, u112, u121 = morphology_masks_from_clc(
        clc_l3,
        urban_111=int(mc["urban_111"]),
        urban_112=int(mc["urban_112"]),
        urban_121=int(mc["urban_121"]),
    )

    rb_cfg = cfg.get("rural_bias") or {}
    rural_bias_arr: np.ndarray | None = None
    rural_idx: tuple[int, ...] = ()
    if bool(rb_cfg.get("enabled", False)):
        src = str(rb_cfg.get("source", "population")).lower()
        rural_min = float(rb_cfg.get("rural_min", 0.4))
        cls_names: list[str] = list(rb_cfg.get("classes") or ("R_FIREPLACE", "R_HEATING_STOVE"))
        rural_idx = tuple(MODEL_CLASSES.index(c) for c in cls_names if c in MODEL_CLASSES)
        pop_path = paths.get("population_tif") or paths.get("ghsl_smod_tif")
        if not pop_path:
            LOG.warning("rural_bias.enabled but no population_tif/ghsl_smod_tif in merged cfg — skipping rural bias")
        else:
            ppop = resolve_path(root, Path(pop_path))
            if not ppop.is_file():
                LOG.warning("rural_bias raster missing %s — skipping", ppop)
            else:
                dens = warp_band_to_ref(ppop, ref, resampling=Resampling.bilinear, band=1)
                rural_bias_arr = rural_bias_from_density(dens, rural_min=rural_min)
                rb_mean = float(np.nanmean(rural_bias_arr))
                frac_low = float(np.mean(rural_bias_arr < 0.6))
                LOG.info(
                    "[other_combustion] rural_bias mean=%.4g frac_lt_0.6=%.4g source=%s",
                    rb_mean,
                    frac_low,
                    src,
                )

    X = build_X_stack(
        R_base,
        C_base,
        u111,
        u112,
        u121,
        mc,
        rural_bias=rural_bias_arr,
        rural_classes=rural_idx if rural_idx else None,
    )
    return {
        "R_base": R_base,
        "C_base": C_base,
        "clc": clc,
        "X": X,
    }
