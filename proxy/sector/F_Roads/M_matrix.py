from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from proxy.core import log
from proxy.core.alias import cams_pollutant_var


def _ef_val(raw: Any) -> float:
    if raw is None or (isinstance(raw, str) and raw.strip().lower() == "missing"):
        return 0.0
    return float(raw)


def load_emission_factor_matrices(
    ef_path: Path,
    pollutants: list[str],
    classes: list[str],
    fuels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaust M[c,f,p] (g/kg fuel) and non-exhaust M[c,p] (g/km/vehicle)."""
    with ef_path.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    tables = doc["tables"]
    pol_keys = [cams_pollutant_var(p) for p in pollutants]
    n_c, n_f, n_p = len(classes), len(fuels), len(pol_keys)
    m_exh = np.zeros((n_c, n_f, n_p), dtype=np.float32)
    m_non = np.zeros((n_c, n_p), dtype=np.float32)
    ci = {c: i for i, c in enumerate(classes)}
    fi = {f: i for i, f in enumerate(fuels)}
    pi = {k: i for i, k in enumerate(pol_keys)}

    for row in tables:
        cls = str(row["vehicle_class"])
        if cls not in ci:
            continue
        ef = row.get("ef") or {}
        row_type = row.get("type")
        if row_type is None:
            fuel = str(row.get("fuel", ""))
            if fuel not in fi:
                continue
            for pk, pj in pi.items():
                if pk in ef:
                    m_exh[ci[cls], fi[fuel], pj] = _ef_val(ef[pk])
        elif row_type in ("tyre_brakes", "road_surface_wear"):
            for pk, pj in pi.items():
                if pk in ef:
                    m_non[ci[cls], pj] += _ef_val(ef[pk])
        elif row_type == "gasoline_evaporation":
            pk = "nmvoc"
            if pk in ef and pk in pi:
                m_non[ci[cls], pi[pk]] += _ef_val(ef[pk])

    log.info(f"F_Roads M_exhaust shape={m_exh.shape} M_non_exhaust shape={m_non.shape}")
    return m_exh, m_non


def log_pi_fleet(
    pi: dict[str, dict[str, float]],
    road_types: list[str],
    classes: list[str],
) -> None:
    log.info("--- F_Roads vehicle fleet Pi (class share per road type, VKM-based) ---")
    hdr = "road_type," + ",".join(classes)
    log.info(hdr)
    for r in road_types:
        row = pi[r]
        vals = ",".join(f"{float(row[c]):.4f}" for c in classes)
        log.info(f"{r},{vals}")


def log_m_by_category(
    m_exh: np.ndarray,
    m_non: np.ndarray,
    classes: list[str],
    fuels: list[str],
    pollutants: list[str],
    f_cats: dict[str, Any],
) -> None:
    pol_keys = [cams_pollutant_var(p) for p in pollutants]
    fi = {f: i for i, f in enumerate(fuels)}

    for cat_name, cat_cfg in f_cats.items():
        if cat_name == "F4":
            log.info(f"--- F_Roads M {cat_name} (non-exhaust, g/km/vehicle) ---")
            log.info("pollutant," + ",".join(classes))
            for pj, pk in enumerate(pol_keys):
                vals = ",".join(f"{float(m_non[ci, pj]):.6g}" for ci in range(len(classes)))
                log.info(f"{pk},{vals}")
            continue
        cat_fuels = [str(f) for f in cat_cfg["fuels"]]
        for f in cat_fuels:
            fi_idx = fi[f]
            log.info(f"--- F_Roads M {cat_name} exhaust {f} (g/kg fuel) ---")
            log.info("pollutant," + ",".join(classes))
            for pj, pk in enumerate(pol_keys):
                vals = ",".join(f"{float(m_exh[ci, fi_idx, pj]):.6g}" for ci in range(len(classes)))
                log.info(f"{pk},{vals}")
