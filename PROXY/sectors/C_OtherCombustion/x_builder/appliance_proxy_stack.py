"""
Seven-band stationary proxy ``X_k = S_k * L_k`` from POP, heat, HDD, GHS-SMOD, CLC (GNFR C).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import MODEL_CLASSES


def _u_rural_mask(
    ghs_smod: np.ndarray,
    clc_l3: np.ndarray,
    *,
    urban_111: int,
    urban_112: int,
    urban_121: int,
    ghs_rural_classes: frozenset[int],
) -> np.ndarray:
    """1 where rint(GHS) in rural set and CLC L3 not in urban morphology codes; NaN-safe."""
    g = np.asarray(ghs_smod, dtype=np.float64)
    c = np.asarray(clc_l3, dtype=np.float64)
    gr = np.rint(g).astype(np.int32)
    rural_codes = np.array(sorted(ghs_rural_classes), dtype=np.int32)
    in_ghs = np.isin(gr, rural_codes)
    finite_c = np.isfinite(c)
    ci = np.where(finite_c, np.rint(c).astype(np.int32), -9999)
    not_urban = (ci != int(urban_111)) & (ci != int(urban_112)) & (ci != int(urban_121))
    out = (in_ghs & finite_c & not_urban).astype(np.float32)
    out[~np.isfinite(g)] = 0.0
    return out


def _other_mask(u111: np.ndarray, u112: np.ndarray, u121: np.ndarray) -> np.ndarray:
    return np.clip(
        1.0 - u111.astype(np.float64) - u112.astype(np.float64) - u121.astype(np.float64),
        0.0,
        1.0,
    ).astype(np.float32)


def _layer_for_name(name: str, H_res: np.ndarray, H_nres: np.ndarray, hdd: np.ndarray) -> np.ndarray:
    u = name.upper()
    if u == "H_RES":
        return H_res
    if u == "H_NRES":
        return H_nres
    if u == "HDD":
        return hdd
    raise ValueError(f"unknown load layer {name!r}")


def _compute_load(
    load_spec: dict[str, Any],
    H_res: np.ndarray,
    H_nres: np.ndarray,
    hdd: np.ndarray,
    eps: float,
) -> np.ndarray:
    t = str(load_spec["type"]).lower()
    if t == "constant":
        v = float(load_spec["value"])
        return np.full_like(H_res, v, dtype=np.float32)
    if t == "variable":
        name = str(load_spec["name"]).upper()
        exp = float(load_spec.get("exponent", 1.0))
        arr = _layer_for_name(name, H_res, H_nres, hdd).astype(np.float64)
        return np.power(np.maximum(arr, 0.0) + eps, exp).astype(np.float32)
    if t == "product":
        out = np.ones_like(H_res, dtype=np.float64)
        for term in load_spec["terms"]:
            name = str(term["name"]).upper()
            exp = float(term["exponent"])
            arr = _layer_for_name(name, H_res, H_nres, hdd).astype(np.float64)
            out *= np.power(np.maximum(arr, 0.0) + eps, exp)
        return out.astype(np.float32)
    raise ValueError(f"unknown load type {t!r}")


def _compute_stock(
    stock_spec: dict[str, Any],
    pop: np.ndarray,
    H_nres: np.ndarray,
    u111: np.ndarray,
    u112: np.ndarray,
    u121: np.ndarray,
    u_rural: np.ndarray,
    other: np.ndarray,
) -> np.ndarray:
    cw = stock_spec["corine_weights"]
    acc = np.zeros_like(u111, dtype=np.float64)
    if "111" in cw:
        acc += float(cw["111"]) * u111.astype(np.float64)
    if "112" in cw:
        acc += float(cw["112"]) * u112.astype(np.float64)
    if "121" in cw:
        acc += float(cw["121"]) * u121.astype(np.float64)
    if "rural_res" in cw:
        acc += float(cw["rural_res"]) * u_rural.astype(np.float64)
    cow = stock_spec.get("commercial_other_weight")
    if cow is not None:
        acc += float(cow) * other.astype(np.float64)

    carrier = str(stock_spec["carrier"]).upper()
    if carrier == "POP":
        acc *= pop.astype(np.float64)
    else:
        acc *= np.maximum(H_nres.astype(np.float64), 0.0)
    return acc.astype(np.float32)


def build_appliance_X_stack(
    *,
    H_res: np.ndarray,
    H_nres: np.ndarray,
    hdd: np.ndarray,
    pop: np.ndarray,
    ghs_smod: np.ndarray,
    clc_l3: np.ndarray,
    u111: np.ndarray,
    u112: np.ndarray,
    u121: np.ndarray,
    urban_111: int,
    urban_112: int,
    urban_121: int,
    appliance_doc: dict[str, Any],
) -> np.ndarray:
    """
    ``X`` float32 (H,W,7) in ``MODEL_CLASSES`` order.
    ``appliance_doc`` validated document from :func:`load_appliance_proxy_from_rules_yaml`.
    """
    h, w = H_res.shape
    k = len(MODEL_CLASSES)
    eps = float(appliance_doc.get("epsilon", 1.0e-12))
    ghs_set = frozenset(int(x) for x in (appliance_doc.get("ghs_rural_classes") or [11, 12, 13]))

    H_res_f = np.asarray(H_res, dtype=np.float32)
    H_nres_f = np.asarray(H_nres, dtype=np.float32)
    hdd_f = np.asarray(hdd, dtype=np.float32)
    pop_f = np.asarray(pop, dtype=np.float32)

    u_rural = _u_rural_mask(
        ghs_smod,
        clc_l3,
        urban_111=urban_111,
        urban_112=urban_112,
        urban_121=urban_121,
        ghs_rural_classes=ghs_set,
    )
    other = _other_mask(u111, u112, u121)

    X = np.zeros((h, w, k), dtype=np.float32)
    for band, cls in enumerate(MODEL_CLASSES):
        spec = appliance_doc[cls]
        S = _compute_stock(
            spec["stock"],
            pop_f,
            H_nres_f,
            u111,
            u112,
            u121,
            u_rural,
            other,
        )
        L = _compute_load(spec["load"], H_res_f, H_nres_f, hdd_f, eps)
        X[:, :, band] = (S * L).astype(np.float32)
    return X
