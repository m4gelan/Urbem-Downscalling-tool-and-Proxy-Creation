from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from PROXY_V2.alpha.resolve_NFR import SectorGroupResolution, resolve_sector_groups_from_block
from PROXY_V2.core import log
from PROXY_V2.core.alias import (
    cams_pollutant_var,
    iso3_from_workbook_country_token,
    workbook_pollutant_label,
)
from PROXY_V2.dataset_loaders.load_alpha_workbook import read_alpha_workbook

def load_alpha_methods_doc(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def _sector_block(doc: dict[str, Any], sector_key: str) -> dict[str, Any]:
    sec = (doc.get("sectors") or {}).get(sector_key) or {}
    return sec if isinstance(sec, dict) else {}


def _norm_yaml_key(k: str) -> str:
    return str(k).strip().lower().replace(".", "_")


def _parse_country_specific_raw(raw: Any) -> frozenset[str]:
    if raw is None:
        return frozenset()
    items: list[str]
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
        items = [p.strip() for p in parts if p.strip()]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(x).strip() for x in raw if str(x).strip()]
    else:
        items = [str(raw).strip()] if str(raw).strip() else []
    out: set[str] = set()
    for tok in items:
        c = str(tok).strip().upper()
        if len(c) == 3 and c.isalpha():
            out.add(c)
    return frozenset(out)


def _parse_method1_pool_settings(merged: dict[str, Any]) -> tuple[str, frozenset[str]]:
    pool_raw = merged.get("pool")
    if pool_raw is None:
        pf = _parse_country_specific_raw(merged.get("pool_from_countries"))
        eu = str(merged.get("eu_pool", "EU27")).strip().upper()
        if eu not in ("EU11", "EU27"):
            raise ValueError(f"invalid eu_pool {eu!r} (use EU11 or EU27)")
        return eu, pf

    if isinstance(pool_raw, str):
        eu = pool_raw.strip().upper()
        if eu not in ("EU11", "EU27"):
            raise ValueError(
                f"method 1 pool string must be EU11 or EU27, got {eu!r}; "
                "use a list of ISO3 codes for a mean-national pool."
            )
        return eu, frozenset()

    if isinstance(pool_raw, (list, tuple, set)):
        eu_fb = str(merged.get("eu_pool", "EU27")).strip().upper()
        if eu_fb not in ("EU11", "EU27"):
            eu_fb = "EU27"
        return eu_fb, _parse_country_specific_raw(pool_raw)

    raise TypeError(f"method 1 pool must be EU11/EU27 string or ISO3 list, got {type(pool_raw).__name__}")


def _method_entry_for_pollutant(
    sector_block: dict[str, Any], pol_yaml_key: str
) -> tuple[int, str | None, frozenset[str], frozenset[str]]:
    pols = sector_block.get("pollutants") or {}
    pk = _norm_yaml_key(pol_yaml_key)
    defaults = sector_block.get("defaults") or {}
    default_method = int(defaults.get("method", 0))

    def _defaults_only() -> tuple[int, str | None, frozenset[str], frozenset[str]]:
        if default_method != 1:
            return default_method, None, frozenset(), frozenset()
        merged = dict(defaults)
        eu, pf = _parse_method1_pool_settings(merged)
        cs = _parse_country_specific_raw(defaults.get("country_specific"))
        return default_method, eu, cs, pf

    for k, spec in pols.items():
        if _norm_yaml_key(str(k)) != pk:
            continue
        if not isinstance(spec, dict):
            return _defaults_only()
        m = int(spec.get("method", default_method))
        if m != 1:
            return m, None, frozenset(), frozenset()
        merged = dict(defaults)
        merged.update(spec)
        eu, pf = _parse_method1_pool_settings(merged)
        cs_src = spec.get("country_specific")
        if cs_src is None:
            cs_src = defaults.get("country_specific")
        cs = _parse_country_specific_raw(cs_src)
        return m, eu, cs, pf

    return _defaults_only()


def _emission_vector(
    long_df: pd.DataFrame,
    country_key: str,
    pollutant_alpha: str,
    group_order: tuple[str, ...],
) -> np.ndarray:
    n_g = len(group_order)
    vec = np.zeros(n_g, dtype=np.float64)
    sub = long_df[
        (long_df["country_iso3"].astype(str).str.upper() == str(country_key).upper())
        & (long_df["pollutant"].astype(str) == str(pollutant_alpha))
    ]
    gid_to_i = {str(g): i for i, g in enumerate(group_order)}
    for _, row in sub.iterrows():
        g = str(row.get("group", ""))
        gi = gid_to_i.get(g)
        if gi is not None:
            vec[gi] += float(row["E"])
    return vec


def _mean_alpha_across_countries(
    long_df: pd.DataFrame,
    pollutant_alpha: str,
    group_order: tuple[str, ...],
    iso_codes: frozenset[str],
) -> np.ndarray | None:
    if not iso_codes:
        return None
    vecs: list[np.ndarray] = []
    for iso in sorted(iso_codes):
        ev = _emission_vector(long_df, iso, pollutant_alpha, group_order)
        s = float(ev.sum())
        if s > 0:
            vecs.append(ev / s)
    if not vecs:
        return None
    arr = np.mean(np.stack(vecs, axis=0), axis=0)
    s2 = float(arr.sum())
    if s2 <= 0:
        return None
    return arr / s2


def _resolve_method1_pool_emissions(
    long_df: pd.DataFrame,
    pollutant_alpha: str,
    group_order: tuple[str, ...],
    eu_pool_key: str,
    pool_from_iso: frozenset[str],
) -> np.ndarray:
    if pool_from_iso:
        ev_n = _mean_alpha_across_countries(long_df, pollutant_alpha, group_order, pool_from_iso)
        if ev_n is not None:
            return ev_n * 1.0e6
        log.warning(
            f"Alpha method 1: pool country list {sorted(pool_from_iso)!r} produced no positive nationals; "
            f"falling back to workbook row {eu_pool_key}"
        )
    return _emission_vector(long_df, eu_pool_key, pollutant_alpha, group_order)


def _method_2_vector(
    sector_block: dict[str, Any],
    pol_yaml_key: str,
    group_order: tuple[str, ...],
) -> np.ndarray:
    pk = _norm_yaml_key(pol_yaml_key)
    m2 = sector_block.get("method_2") or {}
    block: dict[str, float] = {}
    for k, v in m2.items():
        if _norm_yaml_key(str(k)) != pk or not isinstance(v, dict):
            continue
        block = {str(g).strip(): float(x) for g, x in v.items()}
        break
    n = len(group_order)
    vec = np.zeros(n, dtype=np.float64)
    for i, g in enumerate(group_order):
        vec[i] = float(block.get(str(g), 0.0))
    s = float(vec.sum())
    if s <= 0:
        raise ValueError(
            f"method_2 alphas for pollutant {pol_yaml_key!r} are missing or sum to zero "
            f"(keys must match group names: {list(group_order)})."
        )
    return vec / s


def _finalize_alpha_rows(alpha: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64).copy()
    p, g = a.shape
    if p == 0 or g == 0:
        return a
    uniform = np.full(g, 1.0 / float(g), dtype=np.float64)
    for i in range(p):
        row = np.nan_to_num(a[i], nan=0.0, posinf=0.0, neginf=0.0)
        row = np.maximum(row, 0.0)
        total = float(row.sum())
        a[i, :] = row / total if total > tol else uniform
    return a


def _build_group_emissions_long_df(
    workbook_df: pd.DataFrame,
    year: int,
    groups: SectorGroupResolution,
) -> pd.DataFrame:
    df = workbook_df.loc[workbook_df["YEAR"] == int(year)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        sn = str(row.get("SECTOR_NORM", ""))
        gname = groups.group_for_sector_norm(sn)
        if gname is None:
            continue
        try:
            iso3 = iso3_from_workbook_country_token(str(row["COUNTRY"]))
        except ValueError:
            continue
        poll_alpha = str(row["POLLUTANT_ALPHA"])
        if not poll_alpha:
            continue
        val = float(row["TOTAL_VALUE"])
        rows.append({"country_iso3": iso3, "pollutant": poll_alpha, "group": gname, "E": val})
    if not rows:
        return pd.DataFrame(columns=["country_iso3", "pollutant", "group", "E"])
    out = pd.DataFrame(rows)
    return out.groupby(["country_iso3", "pollutant", "group"], as_index=False)["E"].sum()


@dataclass
class AlphaMatrixResult:
    """Per-pollutant split over sector inventory groups (each row sums to 1)."""

    alpha: np.ndarray
    group_names: tuple[str, ...]
    pollutant_labels: list[str]
    methods: np.ndarray
    audit: pd.DataFrame


def compute_sector_alpha_matrix(
    workbook_path: Path,
    alpha_methods_path: Path,
    *,
    sector_key: str,
    year: int,
    country_profile: dict[str, str],
    pollutant_labels: list[str],
) -> AlphaMatrixResult:
    """
    Build ``alpha`` with shape ``(len(pollutant_labels), n_groups)`` for one country and year.

    *sector_key* selects the block under ``alpha_methods.yaml`` → ``sectors`` (e.g. ``J_Waste``).

    Workbook ``COUNTRY`` must use ``Abbreviation`` (e.g. GR); EU pool rows use ``EU27`` / ``EU11``.
    """
    doc = load_alpha_methods_doc(alpha_methods_path)
    sec = _sector_block(doc, sector_key)
    groups = resolve_sector_groups_from_block(sec)
    group_order = groups.group_order
    n_g = len(group_order)
    if n_g == 0:
        raise ValueError(f"{sector_key}: no groups resolved from alpha_methods.yaml")

    abbrev = str(country_profile["Abbreviation"]).strip().upper()
    iso_target = str(country_profile["ISO3"]).strip().upper()

    wb_raw = read_alpha_workbook(workbook_path)
    if wb_raw.empty:
        raise ValueError(f"Alpha workbook is empty: {workbook_path}")

    abbrev_rows = wb_raw.loc[wb_raw["COUNTRY"].astype(str).str.strip().str.upper() == abbrev]
    if abbrev_rows.empty:
        raise ValueError(
            f"No alpha workbook rows for COUNTRY={abbrev!r} (Abbreviation) and any year; "
            f"check country spelling against the sheet."
        )
    if int(year) not in set(abbrev_rows["YEAR"].astype(int).unique()):
        raise ValueError(
            f"No alpha workbook rows for COUNTRY={abbrev!r} and YEAR={year}. "
            f"Available years: {sorted(abbrev_rows['YEAR'].astype(int).unique().tolist())}"
        )

    long_df = _build_group_emissions_long_df(wb_raw, year, groups)
    if long_df.empty:
        raise ValueError(
            f"{sector_key}: no group-mapped emissions for year={year} "
            f"(check NFR codes vs alpha_methods groups for this sector)."
        )

    pol_list = [str(p).strip() for p in pollutant_labels if str(p).strip()]
    n_p = len(pol_list)
    alpha = np.zeros((n_p, n_g), dtype=np.float64)
    methods_arr = np.zeros(n_p, dtype=np.int32)
    audit_rows: list[dict[str, Any]] = []

    uni = np.full(n_g, 1.0 / max(n_g, 1), dtype=np.float64)

    for j, lab in enumerate(pol_list):
        pol_yaml = cams_pollutant_var(lab)
        poll_alpha = workbook_pollutant_label(lab)
        m, eu_pool, cs_set, pf_set = _method_entry_for_pollutant(sec, pol_yaml)

        methods_arr[j] = int(m)
        if m == 4:
            raise ValueError(f"Alpha method 4 is not implemented in PROXY_V2 (sector {sector_key!r})")

        if m == 3:
            alpha[j, :] = uni
            for gi, gname in enumerate(group_order):
                audit_rows.append(
                    {
                        "Poll": poll_alpha,
                        "Group": gname,
                        "country_iso3": iso_target,
                        "E": 0.0,
                        "alpha": float(alpha[j, gi]),
                        "method": m,
                    }
                )
            continue

        if m == 2:
            alpha[j, :] = _method_2_vector(sec, pol_yaml, group_order)
            for gi, gname in enumerate(group_order):
                audit_rows.append(
                    {
                        "Poll": poll_alpha,
                        "Group": gname,
                        "country_iso3": iso_target,
                        "E": 0.0,
                        "alpha": float(alpha[j, gi]),
                        "method": m,
                    }
                )
            continue

        if m == 1:
            pool = str(eu_pool or "EU27").strip().upper()
            if pool not in ("EU11", "EU27"):
                raise ValueError(f"invalid EU pool {pool!r}")
            use_national = iso_target in cs_set
            ev = np.zeros(n_g, dtype=np.float64)
            if use_national:
                ev = _emission_vector(long_df, iso_target, poll_alpha, group_order)
                if float(ev.sum()) <= 0:
                    log.warning(
                        f"{sector_key} alpha: country_specific national emissions zero for "
                        f"{iso_target} {poll_alpha}; falling back to pool {pool}"
                    )
                    ev = _resolve_method1_pool_emissions(
                        long_df, poll_alpha, group_order, pool, pf_set
                    )
            else:
                ev = _resolve_method1_pool_emissions(long_df, poll_alpha, group_order, pool, pf_set)
            if float(ev.sum()) <= 0:
                raise ValueError(
                    f"{sector_key} alpha method 1: emissions sum to zero for {poll_alpha} pool={pool}"
                )
            alpha[j, :] = ev / float(ev.sum())
            for gi, gname in enumerate(group_order):
                audit_rows.append(
                    {
                        "Poll": poll_alpha,
                        "Group": gname,
                        "country_iso3": iso_target,
                        "E": float(ev[gi]),
                        "alpha": float(alpha[j, gi]),
                        "method": m,
                    }
                )
            continue

        if m == 0:
            ev = _emission_vector(long_df, iso_target, poll_alpha, group_order)
            if float(ev.sum()) <= 0:
                raise ValueError(
                    f"{sector_key} alpha method 0: no positive national emissions for "
                    f"{iso_target} {poll_alpha} year={year}"
                )
            alpha[j, :] = ev / float(ev.sum())
            for gi, gname in enumerate(group_order):
                audit_rows.append(
                    {
                        "Poll": poll_alpha,
                        "Group": gname,
                        "country_iso3": iso_target,
                        "E": float(ev[gi]),
                        "alpha": float(alpha[j, gi]),
                        "method": m,
                    }
                )
            continue

        raise ValueError(f"Unknown alpha method {m} for pollutant {lab!r}")

    alpha = _finalize_alpha_rows(alpha)
    audit = pd.DataFrame(audit_rows) if audit_rows else pd.DataFrame()
    return AlphaMatrixResult(
        alpha=alpha,
        group_names=group_order,
        pollutant_labels=pol_list,
        methods=methods_arr,
        audit=audit,
    )


def load_sector_alpha_from_config(
    repo_root: Path,
    sector_cfg: dict[str, Any],
    *,
    sector_key: str,
    year: int,
    country_profile: dict[str, str],
    pollutant_labels: list[str],
) -> AlphaMatrixResult:
    """CEIP α matrix from sector YAML ``filepaths.Alpha`` + ``alpha_methods.yaml``."""
    fp = sector_cfg.get("filepaths") or {}
    alpha_block = fp.get("Alpha") or {}
    methods_rel = alpha_block.get("config")
    xlsx_rel = alpha_block.get("path")
    if not methods_rel or not xlsx_rel:
        raise ValueError("filepaths.Alpha.path and filepaths.Alpha.config required")
    xlsx_path = repo_root / str(xlsx_rel).replace("\\", "/")
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"Alpha workbook not found: {xlsx_path}")
    result = compute_sector_alpha_matrix(
        xlsx_path,
        repo_root / str(methods_rel).replace("\\", "/"),
        sector_key=sector_key,
        year=year,
        country_profile=country_profile,
        pollutant_labels=pollutant_labels,
    )
    log.info(
        f"{sector_key} alpha shape {result.alpha.shape} groups={list(result.group_names)}"
    )
    return result
