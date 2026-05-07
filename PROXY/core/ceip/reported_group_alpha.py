"""CEIP reported emissions -> per-country per-group (or per-subsector) α tensors.

**Role**: build α used by :func:`PROXY.sectors._shared.gnfr_groups.run_gnfr_group_pipeline`
for GNFR **group** sectors (``B_Industry``, ``D_Fugitive``) and by GNFR **E** solvents via
:func:`load_ceip_and_alpha_solvents` (2.D.3 subsectors from ``solvents_subsectors.yaml``).

**Inputs**: merged ``cfg`` (workbook path, profile YAML, ``pollutants``, ordered ids,
optional ``ceip_years`` and pollutant aliases), plus ``iso3_list``.

**Outputs**: ``load_ceip_and_alpha`` returns ``(alpha, fallback_codes, wide)``;
``load_ceip_and_alpha_solvents`` adds a ``meta`` dict.

**Why here**: sector-agnostic CEIP ingestion; only profile layout and ``sector_key`` for YAML
fallback differ between callers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from PROXY.core.alpha import read_alpha_workbook
from PROXY.core.alpha.fallback import AlphaSource, format_provenance, resolve_alpha
from PROXY.core.alpha.reported import (
    normalize_inventory_sector,
    parse_float_or_nan,
    resolve_iso3_reported,
    short_country,
    to_iso3,
)
from PROXY.core.ceip.loader import DEFAULT_GNFR_GROUP_ORDER, remap_legacy_ceip_relpath
from PROXY.core.dataloaders import resolve_path

logger = logging.getLogger(__name__)


_ISO3_TO_SHORT: dict[str, str] = {"GRC": "EL"}


def _short_country(iso3: str) -> str:
    return short_country(iso3, _ISO3_TO_SHORT)


def _apply_fallback_yaml(
    alpha: np.ndarray,
    fallback: np.ndarray,
    iso3_list: list[str],
    pollutants: list[str],
    *,
    sector_key: str,
    group_order: tuple[str, ...],
    focus_country_iso3: str | None = None,
) -> None:
    """Rewrite the uniform-fallback leaves of ``alpha`` using the shared YAML resolver.

    Phase 2.3/2.4: when ``build_alpha_tensor`` marked a (country, pollutant) entry as
    ``fallback_code == 3`` (all-uniform 1/n), consult
    ``PROXY/config/ceip/alpha/fallback/`` via :func:`core.alpha.fallback.resolve_alpha`. If
    the YAML files are empty the resolver returns the same 0.25 values and this is a
    no-op. Any YAML override is applied in place and logged with the provenance source.

    Reported-workbook values (fallback_code 0/1/2) are never touched.

    ``focus_country_iso3`` (e.g. ``GRC``): log non-uniform YAML provenance at INFO only
    for that ISO3; other countries log at DEBUG so full-country CAMS builds stay readable.
    """
    subsectors = list(group_order)
    focus_u = str(focus_country_iso3).strip().upper() if focus_country_iso3 else ""
    n_yaml_rows = 0
    for ri, iso in enumerate(iso3_list):
        if not iso:
            continue
        iso_u = str(iso).strip().upper()
        country_short = _short_country(iso)
        for j, pol in enumerate(pollutants):
            if int(fallback[ri, j]) < 3:
                continue
            n_yaml_rows += 1
            res = resolve_alpha(
                sector=sector_key,
                country=country_short,
                pollutant=str(pol),
                subsectors=subsectors,
            )
            for gi, gid in enumerate(subsectors):
                val = res.values.get(gid)
                if val is None:
                    continue
                alpha[ri, gi, j] = float(val)
            if any(s is not AlphaSource.UNIFORM_FALLBACK for s in res.source.values()):
                msg = "[alpha] sector=%s country=%s pollutant=%s %s"
                args_t = (sector_key, country_short, pol, format_provenance(res))
                if not focus_u or iso_u == focus_u:
                    logger.info(msg, *args_t)
                else:
                    logger.debug(msg, *args_t)
    if focus_u and n_yaml_rows:
        logger.info(
            "%s: CEIP YAML fallback processed for %d (country x pollutant) uniform-tier entries; "
            "[alpha] provenance at INFO for country_iso3=%s only, DEBUG for other countries.",
            sector_key,
            n_yaml_rows,
            focus_u,
        )


def _to_iso3(code: str) -> str:
    return to_iso3(code, {"EL": "GRC", "GR": "GRC"})


def _resolve_iso3_reported(
    country_raw: str,
    cntr_code_to_iso3: dict[str, str],
) -> str | None:
    """
    Map CEIP COUNTRY (often ISO-2 like EL) to ISO-3, same role as in ``alpha._resolve_country_iso3``,
    with a static EU-27 fallback table.
    """
    return resolve_iso3_reported(country_raw, cntr_code_to_iso3)


def _sector_to_group(
    sector_norm: str, tok2g: dict[str, str]
) -> str | None:
    """Match ``ceip_groups.yaml`` tokens (``read_ceip_long`` logic)."""
    tok = str(sector_norm).strip().upper()
    if not tok:
        return None
    for k, g in tok2g.items():
        if tok == k or tok.startswith(k):
            return g
    return None


def read_reported_emissions_fugitive_long(
    xlsx_path: Path,
    *,
    tok2g: dict[str, str],
    pollutant_aliases: dict[str, str],
    years_filter: list[int] | None,
    cntr_code_to_iso3: dict[str, str],
) -> pd.DataFrame:
    """
    Load `alpha.read_alpha_workbook` (``Reported_Emissions_*.xlsx``), map NFR/LRTAP sectors
    in ``ceip_groups.yaml`` to G1..G4, and return **long** means of annual emissions:

    (country, pollutant, group) with ``E`` = mean over years of (sum of ``TOTAL`` in that
    year for that group/pollutant/country), matching the multi-year share logic in
    ``alpha.read_ceip_shares`` (long / triple cases).
    """
    if not xlsx_path.is_file():
        raise FileNotFoundError(
            f"CEIP reported emissions workbook not found: {xlsx_path}"
        )
    raw = read_alpha_workbook(xlsx_path)
    if raw.empty:
        raise ValueError(f"CEIP workbook is empty: {xlsx_path}")
    if years_filter is not None:
        years_set = {int(x) for x in years_filter}
        raw = raw.loc[raw["YEAR"].astype(int).isin(years_set)].copy()
        if raw.empty:
            raise ValueError(
                f"No rows after year filter {sorted(years_set)} in {xlsx_path}"
            )

    if "SECTOR_NORM" not in raw.columns and "SECTOR" in raw.columns:
        raw["SECTOR_NORM"] = raw["SECTOR"].map(normalize_inventory_sector)
    if "SECTOR_NORM" not in raw.columns:
        raise ValueError("Workbook has no SECTOR/SECTOR_NORM after parse.")

    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        cc_raw = r.get("COUNTRY")
        iso3 = _resolve_iso3_reported(str(cc_raw), cntr_code_to_iso3)
        if not iso3 or len(iso3) != 3:
            continue
        sn = str(r.get("SECTOR_NORM", normalize_inventory_sector(r.get("SECTOR", "")))).strip()
        gid = _sector_to_group(sn, tok2g)
        if gid is None:
            continue
        v = _parse_float(r.get("TOTAL_VALUE"))
        if not np.isfinite(v) or v < 0:
            continue
        su = str(r.get("POLLUTANT", "")).strip().upper().replace(".", "_")
        u0 = pollutant_aliases.get(su, su) if su else "TOTAL"
        u = str(u0).strip().upper().replace(".", "_")
        if u == "PM25":
            u = "PM2_5"
        if not u or u in ("", "NAN"):
            u = "TOTAL"
        rows.append(
            {
                "country_iso3": iso3.upper(),
                "pollutant": u,
                "group": gid,
                "E": float(v),
                "YEAR": int(r["YEAR"]),
            }
        )
    if not rows:
        raise ValueError(
            f"No CEIP rows matched ceip_groups.yaml sectors in {xlsx_path}. "
            "Check NFR sector codes in the workbook vs PROXY/config/ceip/profiles/fugitive_groups.yaml."
        )
    dfp = pd.DataFrame(rows)
    # Per calendar year, sum emissions that fall in the same (country, pollutant, group).
    gsum = dfp.groupby(
        ["country_iso3", "YEAR", "pollutant", "group"],
        as_index=False,
    )["E"].sum()
    # Mean of yearly totals = typical-year mass used for group shares.
    m = gsum.groupby(
        ["country_iso3", "pollutant", "group"],
        as_index=False,
    )["E"].mean()
    m = m.rename(columns={"E": "E"})
    return m


def read_reported_emissions_subsector_long(
    xlsx_path: Path,
    *,
    tok2sub: dict[str, str],
    pollutant_aliases: dict[str, str],
    subsectors: list[str],
    years_filter: list[int] | None,
    cntr_code_to_iso3: dict[str, str],
) -> pd.DataFrame:
    """
    ``Reported_Emissions_*.xlsx`` (``read_alpha_workbook`` layout): map NFR rows to
    2.D.3 subsector keys from the solvents profile YAML, return long **mean** annual
    mass ``E`` per (country, pollutant, subsector) — same multi-year averaging as
    :func:`read_reported_emissions_fugitive_long`.
    """
    if not xlsx_path.is_file():
        raise FileNotFoundError(
            f"CEIP reported emissions workbook not found: {xlsx_path}"
        )
    raw = read_alpha_workbook(xlsx_path)
    if raw.empty:
        raise ValueError(f"CEIP workbook is empty: {xlsx_path}")
    if years_filter is not None:
        years_set = {int(x) for x in years_filter}
        raw = raw.loc[raw["YEAR"].astype(int).isin(years_set)].copy()
        if raw.empty:
            raise ValueError(
                f"No rows after year filter {sorted(years_set)} in {xlsx_path}"
            )

    if "SECTOR_NORM" not in raw.columns and "SECTOR" in raw.columns:
        raw["SECTOR_NORM"] = raw["SECTOR"].map(normalize_inventory_sector)
    if "SECTOR_NORM" not in raw.columns:
        raise ValueError("Workbook has no SECTOR/SECTOR_NORM after parse.")

    sub_s = {str(s) for s in subsectors}
    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        cc_raw = r.get("COUNTRY")
        iso3 = _resolve_iso3_reported(str(cc_raw), cntr_code_to_iso3)
        if not iso3 or len(iso3) != 3:
            continue
        sn = str(r.get("SECTOR_NORM", normalize_inventory_sector(r.get("SECTOR", "")))).strip()
        sub = _sector_to_subsector(sn, tok2sub, sub_s)
        if sub is None:
            continue
        v = _parse_float(r.get("TOTAL_VALUE"))
        if not np.isfinite(v) or v < 0:
            continue
        su = str(r.get("POLLUTANT", "")).strip().upper().replace(".", "_")
        u0 = pollutant_aliases.get(su, su) if su else "TOTAL"
        u = str(u0).strip().upper().replace(".", "_")
        if u == "PM25":
            u = "PM2_5"
        if not u or u in ("", "NAN"):
            u = "TOTAL"
        rows.append(
            {
                "country_iso3": iso3.upper(),
                "pollutant": u,
                "subsector": sub,
                "E": float(v),
                "YEAR": int(r["YEAR"]),
            }
        )
    if not rows:
        raise ValueError(
            f"No reported-emission rows matched solvents CEIP profile in {xlsx_path}. "
            "Check NFR codes vs PROXY/config/ceip/profiles/solvents_subsectors.yaml."
        )
    dfp = pd.DataFrame(rows)
    gsum = dfp.groupby(
        ["country_iso3", "YEAR", "pollutant", "subsector"],
        as_index=False,
    )["E"].sum()
    m = gsum.groupby(
        ["country_iso3", "pollutant", "subsector"],
        as_index=False,
    )["E"].mean()
    return m


def _parse_float(v: Any) -> float:
    return parse_float_or_nan(v)


def load_group_mapping(
    yaml_path: Path,
    group_order: tuple[str, ...] | None = None,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Return (token_upper -> group id, raw ``groups`` dict from YAML).

    ``group_order`` restricts which top-level keys are read and fixes iteration order
    (must match ``cfg["group_order"]`` and the alpha tensor axis). Defaults to
    :data:`DEFAULT_GNFR_GROUP_ORDER`.
    """
    import yaml

    order = group_order if group_order is not None else DEFAULT_GNFR_GROUP_ORDER
    with yaml_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    groups: dict[str, Any] = {str(k): v for k, v in dict(raw.get("groups") or {}).items()}
    tok2g: dict[str, str] = {}
    for gid in order:
        spec = groups.get(str(gid))
        if spec is None:
            raise ValueError(
                f"CEIP groups YAML {yaml_path} has no group {gid!r} required by group_order; "
                f"keys present: {sorted(groups)}"
            )
        for sec in spec.get("ceip_sectors", []) or []:
            t = normalize_inventory_sector(str(sec))
            if t:
                tok2g[t] = str(gid)
    return tok2g, groups


def load_subsector_mapping_from_yaml(
    yaml_path: Path,
    subsector_order: tuple[str, ...],
) -> dict[str, str]:
    """
    Build NFR token ``->`` subsector id from ``subsectors:`` in
    ``PROXY/config/ceip/profiles/solvents_subsectors.yaml``.

    ``subsector_order`` matches ``cfg["subsectors"]``; when the same token appears under
    multiple YAML subsectors, the **last** subsector in ``subsector_order`` wins (same
    idea as iterating ``cfg["subsectors"]`` first-to-last over the profile).
    """
    import yaml

    with yaml_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    subs: dict[str, Any] = {
        str(k): v for k, v in dict(raw.get("subsectors") or {}).items()
    }
    tok2sub: dict[str, str] = {}
    for sub_key in subsector_order:
        sk = str(sub_key)
        spec = subs.get(sk)
        if not isinstance(spec, dict):
            continue
        for sec in spec.get("ceip_sectors", []) or []:
            t = normalize_inventory_sector(str(sec))
            if t:
                tok2sub[t] = sk
    return tok2sub


def _sector_to_subsector(
    sector_norm: str,
    tok2sub: dict[str, str],
    subsector_ids: set[str],
) -> str | None:
    """Longest NFR prefix match (same rule as historical ``E_Solvents.ceip``)."""
    sn = str(sector_norm).strip().upper()
    if not sn:
        return None
    for k in sorted(tok2sub.keys(), key=lambda x: -len(str(x))):
        if sn == k or sn.startswith(k):
            sub = tok2sub[k]
            if sub in subsector_ids:
                return sub
    return None


def _pick_col(cols: dict[str, str], *names: str) -> str | None:
    for n in names:
        if n in cols:
            return cols[n]
    return None


def read_ceip_long(
    xlsx_path: Path,
    *,
    sheet: str | None,
    year: int,
    tok2g: dict[str, str],
    pollutant_aliases: dict[str, str],
) -> pd.DataFrame:
    if not xlsx_path.is_file():
        raise FileNotFoundError(f"CEIP fugitive workbook not found: {xlsx_path}")
    try:
        raw = pd.read_excel(xlsx_path, sheet_name=sheet if sheet else 0, header=0, engine="openpyxl")
    except ValueError:
        xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
        raw = pd.read_excel(xlsx_path, sheet_name=xl.sheet_names[0], header=0, engine="openpyxl")
    cols = {str(c).lower().strip(): c for c in raw.columns}
    c_country = _pick_col(cols, "country_code", "country", "iso", "iso3", "iso2")
    c_year = _pick_col(cols, "year", "yr")
    c_sector = _pick_col(cols, "sector", "sector_code", "snap", "nfr", "code", "nace")
    c_total = _pick_col(cols, "total", "emissions", "value", "emission", "gg", "kt")
    c_pol = _pick_col(cols, "pollutant", "compound", "species", "param", "poll")
    if not all([c_country, c_year, c_sector, c_total]):
        raise ValueError(
            f"CEIP fugitive sheet missing columns. Have: {list(raw.columns)}. "
            "Need country, year, sector, total."
        )
    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        try:
            yr = int(float(r[c_year]))
        except (TypeError, ValueError):
            continue
        if yr != int(year):
            continue
        iso3 = _to_iso3(str(r[c_country]))
        if len(iso3) != 3 or not iso3.isalpha():
            continue
        iso3 = iso3.upper()
        tok = normalize_inventory_sector(str(r[c_sector]))
        gid = None
        for k, g in tok2g.items():
            if tok == k or tok.startswith(k):
                gid = g
                break
        if gid is None:
            continue
        v = _parse_float(r[c_total])
        if not np.isfinite(v) or v < 0:
            continue
        if c_pol and str(r[c_pol]).strip():
            pol = str(r[c_pol]).strip().upper().replace(".", "_")
            pol = pollutant_aliases.get(pol, pol)
            if pol == "PM25":
                pol = "PM2_5"
        else:
            pol = "TOTAL"
        rows.append({"country_iso3": iso3, "pollutant": pol, "group": gid, "E": v})
    if not rows:
        raise ValueError(f"No CEIP fugitive rows after filter (year={year}).")
    return pd.DataFrame(rows)


def build_alpha_tensor(
    long_df: pd.DataFrame,
    iso3_list: list[str],
    pollutants: list[str],
    group_order: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Return ``(alpha, fallback_per_pol, wide_debug)``.

    ``alpha`` shape ``(n_iso, n_groups, n_pol)`` with ``n_groups = len(group_order)``.

    Per country and pollutant: if group totals sum to zero, fall back to CEIP
    ``TOTAL`` rows by group for that country; if still zero, fall back to the
    country’s group totals summed across **all** pollutants (excluding the
    synthetic ``TOTAL`` pollutant label); if still zero, use uniform ``1/n_groups``.

    ``fallback_code`` in the wide table is per (country, pollutant): 0 none,
    1 used TOTAL rows, 2 used all-pollutant sum, 3 uniform.
    """
    n_iso = len(iso3_list)
    n_pol = len(pollutants)
    n_g = len(group_order)
    gid_to_i = {str(g): i for i, g in enumerate(group_order)}
    uni = 1.0 / float(max(n_g, 1))

    agg = long_df.groupby(["country_iso3", "pollutant", "group"], as_index=False)["E"].sum()
    tot = long_df[long_df["pollutant"] == "TOTAL"].groupby(["country_iso3", "group"], as_index=False)["E"].sum()
    agg_nop = agg[agg["pollutant"] != "TOTAL"].groupby(["country_iso3", "group"], as_index=False)["E"].sum()

    alpha = np.full((n_iso, n_g, n_pol), uni, dtype=np.float64)
    fallback = np.zeros((n_iso, n_pol), dtype=np.int32)

    pol_norm = agg["pollutant"].str.upper().str.replace(".", "_", regex=False)

    for j, pl in enumerate([p.upper().replace(".", "_") for p in pollutants]):
        sub = agg.loc[pol_norm == pl]
        for ri in range(n_iso):
            iso = str(iso3_list[ri]).strip().upper()
            if not iso:
                continue
            vec = np.zeros(n_g, dtype=np.float64)
            for _, row in sub[sub["country_iso3"] == iso].iterrows():
                gi = gid_to_i.get(str(row["group"]))
                if gi is not None:
                    vec[gi] += float(row["E"])
            if float(vec.sum()) > 0:
                alpha[ri, :, j] = vec / vec.sum()
                continue
            vec2 = np.zeros(n_g, dtype=np.float64)
            for _, row in tot[tot["country_iso3"] == iso].iterrows():
                gi = gid_to_i.get(str(row["group"]))
                if gi is not None:
                    vec2[gi] += float(row["E"])
            if float(vec2.sum()) > 0:
                alpha[ri, :, j] = vec2 / vec2.sum()
                fallback[ri, j] = 1
                continue
            vec3 = np.zeros(n_g, dtype=np.float64)
            for _, row in agg_nop[agg_nop["country_iso3"] == iso].iterrows():
                gi = gid_to_i.get(str(row["group"]))
                if gi is not None:
                    vec3[gi] += float(row["E"])
            if float(vec3.sum()) > 0:
                alpha[ri, :, j] = vec3 / vec3.sum()
                fallback[ri, j] = 2
            else:
                alpha[ri, :, j] = uni
                fallback[ri, j] = 3

    wide_rows: list[dict[str, Any]] = []
    for j, pl in enumerate(pollutants):
        for ri in range(n_iso):
            if not iso3_list[ri]:
                continue
            row_d: dict[str, Any] = {
                "country_iso3": iso3_list[ri],
                "pollutant": pl.lower(),
                "fallback_code": int(fallback[ri, j]),
            }
            for gi, gname in enumerate(group_order):
                row_d[f"alpha_{gname}"] = float(alpha[ri, gi, j])
            wide_rows.append(row_d)
    wide = pd.DataFrame(wide_rows)
    return alpha, fallback, wide


def load_ceip_and_alpha_solvents(
    cfg: dict[str, Any],
    iso3_list: list[str],
    *,
    focus_country_iso3: str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    """
    GNFR E solvents: ``Reported_Emissions_*.xlsx`` + ``solvents_subsectors.yaml`` -> α tensor.

    Same pipeline as :func:`load_ceip_and_alpha` (mean over years, ``build_alpha_tensor``
    TOTAL / cross-pollutant fallbacks, YAML fill for uniform tier), with subsector ids
    from ``cfg["subsectors"]`` instead of G1..G4.

    Returns
    -------
    alpha, fallback, wide, meta
        ``alpha`` shape ``(len(iso3_list), len(subsectors), len(pollutants))``.
    """
    root = Path(cfg["_project_root"])
    paths = cfg["paths"]
    sub_order = tuple(str(s).strip() for s in (cfg.get("subsectors") or ()))
    if not sub_order:
        raise ValueError("cfg['subsectors'] is required for load_ceip_and_alpha_solvents")
    submap_path = resolve_path(
        root,
        remap_legacy_ceip_relpath(str(paths["ceip_subsector_map_yaml"])),
    )
    if not submap_path.is_file():
        raise FileNotFoundError(
            f"Solvents CEIP subsector profile not found: {submap_path}"
        )
    tok2sub = load_subsector_mapping_from_yaml(submap_path, sub_order)
    xlsx = resolve_path(root, str(paths["ceip_workbook"]))
    if not xlsx.is_file():
        raise FileNotFoundError(
            f"CEIP reported emissions workbook not found: {xlsx}"
        )
    yf = cfg.get("ceip_years") or paths.get("ceip_years")
    years_filter: list[int] | None
    if yf is None or (isinstance(yf, list) and len(yf) == 0):
        years_filter = None
    else:
        years_filter = [int(x) for x in (yf if isinstance(yf, (list, tuple)) else [yf])]
    cntr = dict(cfg.get("cntr_code_to_iso3") or paths.get("cntr_code_to_iso3") or {})
    aliases = dict(cfg.get("ceip_pollutant_aliases") or {})
    pollutants = [str(p) for p in cfg["pollutants"]]
    long_df = read_reported_emissions_subsector_long(
        xlsx,
        tok2sub=tok2sub,
        pollutant_aliases=aliases,
        subsectors=list(sub_order),
        years_filter=years_filter,
        cntr_code_to_iso3=cntr,
    )
    long_for_tensor = long_df.rename(columns={"subsector": "group"})
    alpha, fb, wide = build_alpha_tensor(long_for_tensor, iso3_list, pollutants, sub_order)
    _apply_fallback_yaml(
        alpha,
        fb,
        iso3_list,
        pollutants,
        sector_key="E_Solvents",
        group_order=sub_order,
        focus_country_iso3=focus_country_iso3,
    )
    logger.info(
        "E_Solvents CEIP alpha (reported emissions): %s years=%s iso=%d subsectors=%d pollutants=%d",
        xlsx.name,
        "all" if years_filter is None else years_filter,
        len(iso3_list),
        len(sub_order),
        len(pollutants),
    )
    meta: dict[str, Any] = {
        "source": "reported_emissions_eu27",
        "path": str(xlsx),
        "ceip_years": years_filter,
        "ceip_subsector_map_yaml": str(submap_path),
    }
    return alpha, fb, wide, meta


def load_ceip_and_alpha(
    cfg: dict[str, Any],
    iso3_list: list[str],
    *,
    sector_key: str = "D_Fugitive",
    focus_country_iso3: str | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    root: Path = cfg["_project_root"]
    paths = cfg["paths"]
    ypath = resolve_path(
        root, remap_legacy_ceip_relpath(str(paths["ceip_groups_yaml"]))
    )
    group_order = tuple(str(x).strip() for x in (cfg.get("group_order") or ()))
    if not group_order:
        group_order = DEFAULT_GNFR_GROUP_ORDER
    tok2g, _ = load_group_mapping(ypath, group_order)
    xlsx = resolve_path(
        root,
        paths.get("ceip_workbook") or paths.get("ceip_xlsx", ""),
    )
    yf = paths.get("ceip_years")
    years_filter: list[int] | None
    if yf is None or (isinstance(yf, list) and len(yf) == 0):
        years_filter = None
    else:
        years_filter = [int(x) for x in (yf if isinstance(yf, (list, tuple)) else [yf])]
    cntr = dict(cfg.get("cntr_code_to_iso3") or paths.get("cntr_code_to_iso3") or {})
    aliases = dict((cfg.get("ceip_pollutant_aliases") or {}))
    # Prefer reported-emissions workbook (``alpha.read_alpha_workbook``); same layout as I_Offroad.
    if paths.get("use_legacy_fugitive_ceip_xlsx") and paths.get("ceip_xlsx"):
        xlsx_leg = resolve_path(root, paths["ceip_xlsx"])
        year = int(paths.get("ceip_year", 2019))
        sheet = paths.get("ceip_sheet")
        long_df = read_ceip_long(
            xlsx_leg, sheet=sheet, year=year, tok2g=tok2g, pollutant_aliases=aliases
        )
    else:
        long_df = read_reported_emissions_fugitive_long(
            xlsx,
            tok2g=tok2g,
            pollutant_aliases=aliases,
            years_filter=years_filter,
            cntr_code_to_iso3=cntr,
        )
    pollutants = [str(p) for p in cfg["pollutants"]]
    alpha, fb, wide = build_alpha_tensor(long_df, iso3_list, pollutants, group_order)
    _apply_fallback_yaml(
        alpha,
        fb,
        iso3_list,
        pollutants,
        sector_key=sector_key,
        group_order=group_order,
        focus_country_iso3=focus_country_iso3,
    )
    logger.info(
        "%s CEIP alpha (reported emissions): %s years=%s iso=%d groups=%d pollutants=%d",
        sector_key,
        xlsx.name,
        "all" if years_filter is None else years_filter,
        len(iso3_list),
        len(group_order),
        len(pollutants),
    )
    return alpha, fb, wide


__all__ = [
    "build_alpha_tensor",
    "load_ceip_and_alpha",
    "load_ceip_and_alpha_solvents",
    "load_group_mapping",
    "load_subsector_mapping_from_yaml",
    "read_ceip_long",
    "read_reported_emissions_fugitive_long",
    "read_reported_emissions_subsector_long",
]
