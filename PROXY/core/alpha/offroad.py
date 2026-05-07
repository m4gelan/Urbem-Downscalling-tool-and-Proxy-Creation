"""Offroad (GNFR I) subsector-mass helpers and grid-broadcast utilities.

- :func:`build_share_arrays` broadcasts CEIP triples to a fine-grid country-index array.
- :func:`lookup_offroad_triple_for_iso3` resolves a single-country triple with identical
  fallback rules to :func:`build_share_arrays`.
- :func:`load_offroad_mass_fractions_from_alpha_csv` and
  :func:`resolve_subsector_emission_masses` read the legacy ``alpha_values.csv`` layout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .aliases import _norm_pol, normalize_country_token
from .fallback import AlphaSource, format_provenance, resolve_alpha

_ISO3_TO_SHORT_OFFROAD: dict[str, str] = {"GRC": "EL"}
_OFFROAD_SUBSECTORS: tuple[str, ...] = ("rail", "pipeline", "nonroad")


def _short_country_offroad(iso3: str) -> str:
    c = str(iso3 or "").strip().upper()
    return _ISO3_TO_SHORT_OFFROAD.get(c, c)


def build_share_arrays(
    country_idx: np.ndarray,
    idx_to_iso: dict[int, str],
    shares: dict[str, dict[str, tuple[float, float, float]]],
    pollutant: str,
    fallback_iso: str,
    *,
    default_triple: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast ``(s_rail, s_pipe, s_nr)`` to grid using ``country_idx``."""
    pk = _norm_pol(pollutant)
    table = shares.get(pk, {})
    sr = np.zeros_like(country_idx, dtype=np.float32)
    sp = np.zeros_like(country_idx, dtype=np.float32)
    sn = np.zeros_like(country_idx, dtype=np.float32)
    fb = fallback_iso.strip().upper()

    def _fallback_trip() -> tuple[float, float, float]:
        if table.get(fb) is not None:
            return table[fb]
        if table:
            return next(iter(table.values()))
        return default_triple

    trip_fallback = _fallback_trip()

    for k, iso in idx_to_iso.items():
        if k <= 0:
            continue
        iso_u = str(iso).strip().upper()
        trip = table.get(iso_u) or table.get(fb)
        if trip is None:
            trip = trip_fallback if table else default_triple
        sr[country_idx == k] = np.float32(trip[0])
        sp[country_idx == k] = np.float32(trip[1])
        sn[country_idx == k] = np.float32(trip[2])

    mask = country_idx == 0
    trip0 = table.get(fb)
    if trip0 is None:
        trip0 = trip_fallback if table else default_triple
    sr[mask] = np.float32(trip0[0])
    sp[mask] = np.float32(trip0[1])
    sn[mask] = np.float32(trip0[2])
    return sr, sp, sn


def lookup_offroad_triple_for_iso3(
    shares: dict[str, dict[str, tuple[float, float, float]]],
    pollutant: str,
    preferred_iso3: str,
    *,
    default_triple: tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3),
) -> tuple[float, float, float]:
    """Resolve ``(s_rail, s_pipeline, s_nonroad)`` for one ISO3 code with identical
    fallback rules to :func:`build_share_arrays` (``country_idx == 0`` branch).
    """
    pk = _norm_pol(pollutant)
    table = shares.get(pk, {})
    fb = str(preferred_iso3).strip().upper()

    def _fallback_trip() -> tuple[float, float, float]:
        if table.get(fb) is not None:
            return table[fb]
        if table:
            return next(iter(table.values()))
        return default_triple

    trip_fallback = _fallback_trip()
    trip0 = table.get(fb)
    if trip0 is None:
        trip0 = trip_fallback if table else default_triple
    return (float(trip0[0]), float(trip0[1]), float(trip0[2]))


def resolve_offroad_triple_with_yaml(
    triple: tuple[float, float, float],
    default_triple: tuple[float, float, float],
    *,
    pollutant: str,
    iso3: str,
    logger: Any | None = None,
) -> tuple[float, float, float]:
    """Apply configured I_Offroad alpha fallback only when CEIP yielded the default triple."""
    if not all(abs(a - b) < 1e-12 for a, b in zip(triple, default_triple, strict=False)):
        return triple
    country = _short_country_offroad(iso3)
    res = resolve_alpha(
        sector="I_Offroad",
        country=country,
        pollutant=str(pollutant),
        subsectors=list(_OFFROAD_SUBSECTORS),
    )
    has_override = any(s is AlphaSource.CONFIG_COUNTRY_OVERRIDE for s in res.source.values())
    if not has_override:
        return triple
    new_triple = (
        float(res.values.get("rail", triple[0])),
        float(res.values.get("pipeline", triple[1])),
        float(res.values.get("nonroad", triple[2])),
    )
    if logger is not None:
        logger.info(
            "[alpha] sector=I_Offroad country=%s pollutant=%s %s",
            country,
            pollutant,
            format_provenance(res),
        )
    return new_triple


def apply_offroad_yaml_overrides(
    share_dict: dict[str, dict[str, tuple[float, float, float]]],
    *,
    pollutants: list[str],
    isos: list[str],
    default_triple: tuple[float, float, float],
    logger: Any | None = None,
) -> None:
    """Patch default or missing CEIP triples in-place using I_Offroad YAML alpha fallbacks."""
    for pol in pollutants:
        table = share_dict.setdefault(pol, {})
        for iso3 in isos:
            iso_u = str(iso3).strip().upper()
            if not iso_u:
                continue
            current = table.get(iso_u)
            looks_default = current is None or all(
                abs(a - b) < 1e-12 for a, b in zip(current, default_triple, strict=False)
            )
            if not looks_default:
                continue
            country = _short_country_offroad(iso_u)
            res = resolve_alpha(
                sector="I_Offroad",
                country=country,
                pollutant=str(pol),
                subsectors=list(_OFFROAD_SUBSECTORS),
            )
            new_triple = (
                float(res.values.get("rail", default_triple[0])),
                float(res.values.get("pipeline", default_triple[1])),
                float(res.values.get("nonroad", default_triple[2])),
            )
            if any(s is AlphaSource.CONFIG_COUNTRY_OVERRIDE for s in res.source.values()):
                table[iso_u] = new_triple
                if logger is not None:
                    logger.info(
                        "[alpha] sector=I_Offroad country=%s pollutant=%s %s",
                        country,
                        pol,
                        format_provenance(res),
                    )


def load_offroad_mass_fractions_from_alpha_csv(
    csv_path: Path,
    *,
    country: str,
    subsector_keys: list[str],
    pollutant: str | None,
    gnfr_sector: str = "I_Offroad",
) -> tuple[dict[str, float], dict[str, Any]]:
    """Read ``alpha`` column for ``I_Offroad`` rows and normalize to mass fractions."""
    meta: dict[str, Any] = {"csv_path": str(csv_path), "status": "ok"}
    if not csv_path.is_file():
        meta["status"] = "missing_file"
        return {}, meta

    cc = normalize_country_token(country)
    df = pd.read_csv(csv_path)
    if df.empty or "gnfr_sector" not in df.columns:
        meta["status"] = "empty_or_bad_schema"
        return {}, meta

    df = df.copy()
    df["gnfr_sector"] = df["gnfr_sector"].astype(str).str.strip()
    df["subsector"] = df["subsector"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip().str.upper()
    if "pollutant" in df.columns:
        df["pollutant"] = df["pollutant"].astype(str).str.strip().str.upper()

    sub = df[
        (df["country"] == cc)
        & (df["gnfr_sector"] == str(gnfr_sector).strip())
        & (df["subsector"].isin(list(subsector_keys)))
    ]
    if pollutant and "pollutant" in sub.columns:
        pol = str(pollutant).strip().upper()
        sub_pol = sub[sub["pollutant"] == pol]
        if not sub_pol.empty:
            sub = sub_pol
        else:
            meta["pollutant_filter_no_rows"] = pol

    if sub.empty:
        meta["status"] = "no_matching_rows"
        return {}, meta

    raw: dict[str, list[float]] = {k: [] for k in subsector_keys}
    for _, row in sub.iterrows():
        sk = str(row["subsector"])
        if sk not in raw:
            continue
        try:
            raw[sk].append(float(row["alpha"]))
        except (TypeError, ValueError):
            continue

    masses: dict[str, float] = {}
    for sk in subsector_keys:
        vals = raw.get(sk) or []
        if not vals:
            continue
        masses[sk] = float(np.mean(np.maximum(np.asarray(vals, dtype=np.float64), 0.0)))

    if not masses:
        meta["status"] = "no_alpha_values"
        return {}, meta

    s = float(sum(max(v, 0.0) for v in masses.values()))
    if s <= 0:
        meta["status"] = "nonpositive_sum"
        return {}, meta

    out = {k: float(max(v, 0.0)) / s for k, v in masses.items()}
    meta["pollutant_used"] = str(pollutant).upper() if pollutant else "(any or first block)"
    return out, meta


def resolve_subsector_emission_masses(
    rows: list[dict[str, Any]],
    area_proxy: dict[str, Any],
    *,
    repo_root: Path,
    country: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach ``emission_mass_fraction`` to each activity row: YAML if set, else alpha CSV, else equal."""
    meta: dict[str, Any] = {"mass_source": "yaml"}
    keys = [str(r["subsector_key"]) for r in rows]

    if all(r.get("emission_mass_fraction") is not None for r in rows):
        return [dict(r) for r in rows], meta

    if not bool(area_proxy.get("use_alpha_emission_split", True)):
        n = max(len(rows), 1)
        meta["mass_source"] = "equal"
        return [{**dict(r), "emission_mass_fraction": 1.0 / n} for r in rows], meta

    rel = str(area_proxy.get("alpha_values_csv", "OUTPUT/Proxy_weights/_alpha/alpha_values.csv"))
    csv_path = Path(rel) if Path(rel).is_absolute() else repo_root / rel
    pol = area_proxy.get("alpha_pollutant")
    if pol is not None:
        pol = str(pol).strip()

    m, am = load_offroad_mass_fractions_from_alpha_csv(
        csv_path,
        country=country,
        subsector_keys=keys,
        pollutant=pol,
    )
    meta["alpha_lookup"] = am
    if not m or float(sum(m.values())) <= 0.0:
        n = max(len(rows), 1)
        meta["mass_source"] = "equal_fallback_no_alpha"
        return [{**dict(r), "emission_mass_fraction": 1.0 / n} for r in rows], meta

    vec = np.array([float(m.get(k, 0.0)) for k in keys], dtype=np.float64)
    tot = float(vec.sum())
    if tot <= 0:
        n = max(len(rows), 1)
        meta["mass_source"] = "equal_fallback_alpha_zero"
        return [{**dict(r), "emission_mass_fraction": 1.0 / n} for r in rows], meta
    vec /= tot
    meta["mass_source"] = "alpha_csv"
    return [
        {**dict(r), "emission_mass_fraction": float(vec[i])} for i, r in enumerate(rows)
    ], meta
