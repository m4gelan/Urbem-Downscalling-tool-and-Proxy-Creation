"""Unified CEIP alpha computation by explicit **method** (0–4) per sector × pollutant.

See ``PROXY/config/ceip/alpha/alpha_methods.yaml`` for configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from PROXY.core.alpha.aliases import norm_pollutant_key
from PROXY.core.alpha.reported import EU_AGGREGATE_CODES, to_iso3

logger = logging.getLogger(__name__)

# Common non-ISO workbook / shorthand tokens -> ISO-3166 alpha-3 (for ``country_specific``).
_ISO3_TYPOS: dict[str, str] = {
    "GER": "DEU",
    "SPA": "ESP",
    "NED": "NLD",
    "GRE": "GRC",
    "UK": "GBR",
}

DEFAULT_METHODS_REL = Path("PROXY") / "config" / "ceip" / "alpha" / "alpha_methods.yaml"


def _project_root_from_cfg(cfg: dict[str, Any]) -> Path:
    r = cfg.get("_project_root")
    if r is not None:
        return Path(r).resolve()
    return Path(__file__).resolve().parents[2]


def load_alpha_methods_doc(root: Path) -> dict[str, Any]:
    p = root / DEFAULT_METHODS_REL
    if not p.is_file():
        return {}
    with p.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw if isinstance(raw, dict) else {}


def _config_path_for_errors(doc: dict[str, Any], root: Path) -> str:
    rel = doc.get("config_file") or str(DEFAULT_METHODS_REL.as_posix())
    return str((root / rel).resolve()) if not Path(rel).is_absolute() else rel


def _sector_block(doc: dict[str, Any], sector_key: str) -> dict[str, Any]:
    sec = (doc.get("sectors") or {}).get(sector_key) or {}
    return sec if isinstance(sec, dict) else {}


def _normalize_country_specific_token(tok: str) -> str:
    """Map shorthand tokens (``GER``, ``SPA``, …) to ISO-3166 alpha-3; ignore EU pools."""
    c = str(tok).strip().upper()
    if not c:
        return ""
    if c in EU_AGGREGATE_CODES:
        logger.warning(
            "Ignoring country_specific token %r (EU11/EU27 belong under ``pool:``, not country_specific)",
            tok,
        )
        return ""
    if c in _ISO3_TYPOS:
        return _ISO3_TYPOS[c]
    if len(c) == 3 and c.isalpha():
        return c
    mapped = to_iso3(c)
    if len(mapped) == 3 and str(mapped).isalpha():
        return str(mapped).strip().upper()
    logger.warning("Could not normalize country_specific token %r to ISO3; ignoring", tok)
    return ""


def _parse_country_specific_raw(raw: Any) -> frozenset[str]:
    """YAML ``country_specific``: list of codes, or comma/semicolon-separated string."""
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
        norm = _normalize_country_specific_token(tok)
        if norm:
            out.add(norm)
    return frozenset(out)


def _mean_alpha_across_countries(
    long_df: pd.DataFrame,
    pol_upper: str,
    group_order: tuple[str, ...],
    iso_codes: frozenset[str],
    *,
    sector_key: str,
    pollutant_label: str,
) -> np.ndarray | None:
    """
    Mean of **normalized** national split vectors (equal weight per ISO).

    Countries with zero reported total for this pollutant are skipped. Returns ``None``
    if no country contributes.
    """
    if not iso_codes:
        return None
    vecs: list[np.ndarray] = []
    skipped: list[str] = []
    for iso in sorted(iso_codes):
        ev = _emission_vector(long_df, iso, pol_upper, group_order)
        s = float(ev.sum())
        if s > 0:
            vecs.append(ev / s)
        else:
            skipped.append(iso)
    if skipped:
        logger.warning(
            "[alpha_method] %s poll %s: pool country list skipping zero-emission ISOs: %s",
            sector_key,
            pollutant_label,
            ", ".join(skipped),
        )
    if not vecs:
        return None
    arr = np.mean(np.stack(vecs, axis=0), axis=0)
    s2 = float(arr.sum())
    if s2 <= 0:
        return None
    return arr / s2


def _parse_method1_pool_settings(merged: dict[str, Any]) -> tuple[str, frozenset[str]]:
    """
    Method 1 pool configuration from merged defaults + pollutant block.

    **Unified** ``pool`` key:

    - ``pool: EU27`` or ``pool: EU11`` — workbook aggregate row for that code.
    - ``pool: [FRA, DEU, …]`` — equal-weight mean of national splits; fallback workbook row
      from legacy ``eu_pool`` if present, else **EU27**.

    **Legacy** (still supported): ``eu_pool`` without ``pool``, and/or ``pool_from_countries``.
    """
    pool_raw = merged.get("pool")
    if pool_raw is None:
        pf = _parse_country_specific_raw(merged.get("pool_from_countries"))
        eu = str(merged.get("eu_pool", "EU27")).strip().upper()
        if eu not in ("EU11", "EU27"):
            raise ValueError(
                f"invalid eu_pool {eu!r} (use EU11 or EU27); prefer ``pool: EU27`` or ``pool: [ISO3, …]``"
            )
        return eu, pf

    if isinstance(pool_raw, str):
        eu = pool_raw.strip().upper()
        if eu not in ("EU11", "EU27"):
            raise ValueError(
                f"method 1 ``pool`` string must be EU11 or EU27, got {eu!r}; "
                "use a list of ISO codes for a mean-national pool."
            )
        return eu, frozenset()

    if isinstance(pool_raw, (list, tuple, set)):
        eu_fb = str(merged.get("eu_pool", "EU27")).strip().upper()
        if eu_fb not in ("EU11", "EU27"):
            eu_fb = "EU27"
        return eu_fb, _parse_country_specific_raw(pool_raw)

    raise TypeError(
        f"method 1 ``pool`` must be EU11/EU27 string or a list of countries, got {type(pool_raw).__name__}"
    )


def _method_entry_for_pollutant(
    sector_block: dict[str, Any], pollutant: str
) -> tuple[int, str | None, frozenset[str], frozenset[str]]:
    pols = sector_block.get("pollutants") or {}
    pk = norm_pollutant_key(pollutant)
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
        if norm_pollutant_key(str(k)) == pk:
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


def _resolve_method1_pool_emissions(
    long_df: pd.DataFrame,
    pol_upper: str,
    group_order: tuple[str, ...],
    eu_pool_key: str,
    pool_from_iso: frozenset[str],
    *,
    sector_key: str,
    pollutant_label: str,
    config_ref: str,
) -> np.ndarray:
    """
    Pool vector for method 1 when **not** using pure national rows.

    If ``pool_from_iso`` is non-empty, use mean of normalized national splits for those
    ISOs (equal weight per country). Otherwise use the EU11/EU27 aggregate workbook row.
    """
    if pool_from_iso:
        ev_n = _mean_alpha_across_countries(
            long_df,
            pol_upper,
            group_order,
            pool_from_iso,
            sector_key=sector_key,
            pollutant_label=pollutant_label,
        )
        if ev_n is not None:
            return ev_n * 1.0e6
        logger.warning(
            "[alpha_method] %s poll %s: pool country list %s produced no positive nationals; "
            "falling back to workbook row %s (see %s)",
            sector_key,
            pollutant_label,
            ", ".join(sorted(pool_from_iso)),
            eu_pool_key,
            config_ref,
        )
    return _emission_vector(long_df, eu_pool_key, pol_upper, group_order)


def _method_2_vector(
    sector_block: dict[str, Any],
    pollutant: str,
    group_order: tuple[str, ...],
) -> np.ndarray:
    pk = norm_pollutant_key(pollutant)
    m2 = sector_block.get("method_2") or {}
    block: dict[str, Any] = {}
    for k, v in m2.items():
        if norm_pollutant_key(str(k)) == pk and isinstance(v, dict):
            block = {str(g).strip(): float(x) for g, x in v.items()}
            break
    n = len(group_order)
    vec = np.zeros(n, dtype=np.float64)
    for i, g in enumerate(group_order):
        vec[i] = float(block.get(str(g), block.get(g.upper(), 0.0)))
    s = float(vec.sum())
    if s <= 0:
        raise ValueError(
            f"method_2 alphas for pollutant {pollutant!r} are missing or sum to zero "
            f"(sector block method_2)."
        )
    return vec / s


def _emission_vector(
    long_df: pd.DataFrame,
    country_key: str,
    pollutant_upper: str,
    group_order: tuple[str, ...],
) -> np.ndarray:
    n_g = len(group_order)
    vec = np.zeros(n_g, dtype=np.float64)
    pol_norm = pollutant_upper.upper().replace(".", "_")
    sub = long_df[
        (long_df["country_iso3"].astype(str).str.upper() == str(country_key).upper())
        & (long_df["pollutant"].astype(str).str.upper().str.replace(".", "_", regex=False) == pol_norm)
    ]
    gid_to_i = {str(g): i for i, g in enumerate(group_order)}
    col_group = "group" if "group" in sub.columns else "subsector"
    for _, row in sub.iterrows():
        g = str(row.get(col_group, ""))
        gi = gid_to_i.get(g)
        if gi is not None:
            vec[gi] += float(row["E"])
    return vec


def _log_zero_audit_table(
    *,
    sector_key: str,
    gnfr_letter: str,
    pollutant: str,
    group_order: tuple[str, ...],
) -> None:
    hdr = "Poll\tGroup\tGNFR sector\tEmissions total\talpha value\tmethod"
    lines = [hdr]
    for g in group_order:
        lines.append(f"{pollutant}\t{g}\t{gnfr_letter}\t0.0\t0.0\t(error)")
    logger.error("[alpha_method] %s failure audit:\n%s", sector_key, "\n".join(lines))


def _letter_from_sector(sector_key: str) -> str:
    mapping = {
        "A_PublicPower": "A",
        "B_Industry": "B",
        "C_OtherCombustion": "C",
        "D_Fugitive": "D",
        "E_Solvents": "E",
        "G_Shipping": "G",
        "H_Aviation": "H",
        "I_Offroad": "I",
        "J_Waste": "J",
        "K_Agriculture": "K",
    }
    return mapping.get(sector_key, "")


def build_alpha_tensor_methods(
    long_df: pd.DataFrame,
    iso3_list: list[str],
    pollutants: list[str],
    group_order: tuple[str, ...],
    *,
    sector_key: str,
    cfg: dict[str, Any],
    gnfr_letter: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Build alpha tensor using ``alpha_methods.yaml`` per pollutant.

    For unit tests, ``cfg['_alpha_methods_doc_override']`` may supply an in-memory YAML dict
    instead of reading ``alpha_methods.yaml`` from disk.

    Returns
    -------
    alpha
        Shape ``(n_iso, n_groups, n_pol)``.
    wide
        Per (country_iso3, pollutant): ``method``, ``alpha_<g>``.
    audit
        Long table: Poll, Group, GNFR sector, Emissions total, alpha value, method.
    """
    root = _project_root_from_cfg(cfg)
    doc_raw = cfg.get("_alpha_methods_doc_override")
    if doc_raw is None:
        doc = load_alpha_methods_doc(root)
    else:
        doc = doc_raw
    if not isinstance(doc, dict):
        doc = {}
    sec_block = _sector_block(doc, sector_key)
    config_ref = _config_path_for_errors(doc, root)
    letter = str(gnfr_letter or sec_block.get("gnfr_letter") or _letter_from_sector(sector_key)).strip().upper()

    n_iso = len(iso3_list)
    n_pol = len(pollutants)
    n_g = len(group_order)
    uni = np.full(n_g, 1.0 / max(n_g, 1), dtype=np.float64)

    methods_for_pol: list[int] = []
    eu_pools: list[str | None] = []
    country_specific_sets: list[frozenset[str]] = []
    pool_from_sets: list[frozenset[str]] = []
    for p in pollutants:
        m, eu, cs, pf = _method_entry_for_pollutant(sec_block, p)
        methods_for_pol.append(m)
        eu_pools.append(eu)
        country_specific_sets.append(cs)
        pool_from_sets.append(pf)

    alpha = np.zeros((n_iso, n_g, n_pol), dtype=np.float64)
    method0_alpha: dict[tuple[str, str], np.ndarray] = {}

    for j, pl in enumerate(pollutants):
        pol_upper = str(pl).upper().replace(".", "_")
        m = methods_for_pol[j]
        eu_pool = eu_pools[j]

        for ri in range(n_iso):
            iso = str(iso3_list[ri]).strip().upper()
            if not iso:
                continue

            if m == 4:
                continue

            if m == 3:
                alpha[ri, :, j] = uni
                continue

            if m == 2:
                alpha[ri, :, j] = _method_2_vector(sec_block, pl, group_order)
                continue

            if m == 1:
                pool = str(eu_pool or "EU27").strip().upper()
                if pool not in ("EU11", "EU27"):
                    raise ValueError(f"invalid eu_pool {pool!r} (use EU11 or EU27)")
                cs_set = country_specific_sets[j]
                pf_set = pool_from_sets[j]
                use_national = iso in cs_set
                ev = np.zeros(n_g, dtype=np.float64)
                if use_national:
                    ev = _emission_vector(long_df, iso, pol_upper, group_order)
                    if float(ev.sum()) <= 0:
                        logger.warning(
                            "[alpha_method] %s poll %s country %s: country_specific national "
                            "emissions sum to zero; falling back to pool",
                            sector_key,
                            pol_upper,
                            iso,
                        )
                        ev = _resolve_method1_pool_emissions(
                            long_df,
                            pol_upper,
                            group_order,
                            pool,
                            pf_set,
                            sector_key=sector_key,
                            pollutant_label=str(pl),
                            config_ref=config_ref,
                        )
                else:
                    ev = _resolve_method1_pool_emissions(
                        long_df,
                        pol_upper,
                        group_order,
                        pool,
                        pf_set,
                        sector_key=sector_key,
                        pollutant_label=str(pl),
                        config_ref=config_ref,
                    )
                if float(ev.sum()) <= 0:
                    _log_zero_audit_table(
                        sector_key=sector_key,
                        gnfr_letter=letter,
                        pollutant=pol_upper,
                        group_order=group_order,
                    )
                    raise ValueError(
                        f"For poll {pol_upper} (method 1, pool {pool}) emissions sum to zero; "
                        f"change the method in {config_ref}"
                    )
                alpha[ri, :, j] = ev / float(ev.sum())
                continue

            if m == 0:
                ev = _emission_vector(long_df, iso, pol_upper, group_order)
                if float(ev.sum()) <= 0:
                    _log_zero_audit_table(
                        sector_key=sector_key,
                        gnfr_letter=letter,
                        pollutant=pol_upper,
                        group_order=group_order,
                    )
                    raise ValueError(
                        f"For poll {pol_upper} country {iso} method 0 has no positive emissions; "
                        f"change the method in {config_ref}"
                    )
                a = ev / float(ev.sum())
                alpha[ri, :, j] = a
                method0_alpha[(iso, norm_pollutant_key(pl))] = a.copy()
                continue

            raise ValueError(f"Unknown method {m} for {sector_key} pollutant {pl}")

    for j, pl in enumerate(pollutants):
        if methods_for_pol[j] != 4:
            continue
        pol_upper = str(pl).upper().replace(".", "_")
        for ri in range(n_iso):
            iso = str(iso3_list[ri]).strip().upper()
            if not iso:
                continue
            vecs: list[np.ndarray] = []
            for jj, pp in enumerate(pollutants):
                if methods_for_pol[jj] != 0:
                    continue
                key = (iso, norm_pollutant_key(pp))
                if key in method0_alpha:
                    vecs.append(method0_alpha[key])
            if not vecs:
                _log_zero_audit_table(
                    sector_key=sector_key,
                    gnfr_letter=letter,
                    pollutant=pol_upper,
                    group_order=group_order,
                )
                raise ValueError(
                    f"For poll {pol_upper} method 4 needs at least one pollutant with method 0 "
                    f"for country {iso}; change the method in {config_ref}"
                )
            arr = np.stack(vecs, axis=0)
            a = np.mean(arr, axis=0)
            alpha[ri, :, j] = a / float(np.sum(a))

    wide = _alpha_to_wide(alpha, iso3_list, pollutants, group_order, methods_for_pol)
    audit = _build_audit_table(
        alpha,
        long_df,
        iso3_list,
        pollutants,
        group_order,
        methods_for_pol,
        eu_pools,
        country_specific_sets,
        pool_from_sets,
        letter,
        sector_key,
        config_ref,
    )
    return alpha.astype(np.float64), wide, audit


def _alpha_to_wide(
    alpha: np.ndarray,
    iso3_list: list[str],
    pollutants: list[str],
    group_order: tuple[str, ...],
    methods_for_pol: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for j, pl in enumerate(pollutants):
        for ri, iso in enumerate(iso3_list):
            if not iso:
                continue
            d: dict[str, Any] = {
                "country_iso3": str(iso).strip().upper(),
                "pollutant": str(pl).lower(),
                "method": int(methods_for_pol[j]),
            }
            for gi, g in enumerate(group_order):
                d[f"alpha_{g}"] = float(alpha[ri, gi, j])
            rows.append(d)
    return pd.DataFrame(rows)


def _build_audit_table(
    alpha: np.ndarray,
    long_df: pd.DataFrame,
    iso3_list: list[str],
    pollutants: list[str],
    group_order: tuple[str, ...],
    methods_for_pol: list[int],
    eu_pools: list[str | None],
    country_specific_sets: list[frozenset[str]],
    pool_from_sets: list[frozenset[str]],
    letter: str,
    sector_key: str,
    config_ref: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for j, pl in enumerate(pollutants):
        pol_upper = str(pl).upper().replace(".", "_")
        m = methods_for_pol[j]
        eu_pool = eu_pools[j]
        cs_set = country_specific_sets[j]
        pf_set = pool_from_sets[j]
        for ri, iso in enumerate(iso3_list):
            if not iso:
                continue
            ev = np.zeros(len(group_order), dtype=np.float64)
            if m == 0:
                ev = _emission_vector(long_df, iso, pol_upper, group_order)
            elif m == 1:
                pool = str(eu_pool or "EU27").strip().upper()
                if iso in cs_set:
                    ev_nat = _emission_vector(long_df, iso, pol_upper, group_order)
                    if float(ev_nat.sum()) > 0:
                        ev = ev_nat
                    else:
                        ev = _resolve_method1_pool_emissions(
                            long_df,
                            pol_upper,
                            group_order,
                            pool,
                            pf_set,
                            sector_key=sector_key,
                            pollutant_label=str(pl),
                            config_ref=config_ref,
                        )
                else:
                    ev = _resolve_method1_pool_emissions(
                        long_df,
                        pol_upper,
                        group_order,
                        pool,
                        pf_set,
                        sector_key=sector_key,
                        pollutant_label=str(pl),
                        config_ref=config_ref,
                    )

            for gi, gname in enumerate(group_order):
                rows.append(
                    {
                        "Poll": pol_upper,
                        "Group": str(gname),
                        "GNFR sector": letter,
                        "Emissions total": float(ev[gi]),
                        "alpha value": float(alpha[ri, gi, j]),
                        "method": int(m),
                    }
                )
    return pd.DataFrame(rows)


def method_matrix_from_wide(
    wide: pd.DataFrame,
    iso3_list: list[str],
    pollutants: list[str],
) -> np.ndarray:
    """Build ``(n_iso, n_pol)`` integer matrix of **method** codes from ``wide``."""
    n_iso = len(iso3_list)
    n_pol = len(pollutants)
    mat = np.zeros((n_iso, n_pol), dtype=np.int32)
    iso_idx = {str(iso).strip().upper(): i for i, iso in enumerate(iso3_list) if iso}
    pol_lower = [str(p).strip().lower() for p in pollutants]
    if wide.empty or "method" not in wide.columns:
        return mat
    for _, row in wide.iterrows():
        ri = iso_idx.get(str(row.get("country_iso3", "")).strip().upper())
        pl = str(row.get("pollutant", "")).strip().lower()
        if ri is None or pl not in pol_lower:
            continue
        j = pol_lower.index(pl)
        mat[ri, j] = int(row["method"])
    return mat


__all__ = [
    "build_alpha_tensor_methods",
    "load_alpha_methods_doc",
    "DEFAULT_METHODS_REL",
    "_emission_vector",
    "method_matrix_from_wide",
]
