"""Unified alpha fallback resolver.

For NH3/CH4/CO2 in particular, reported country-specific alpha values are often
incomplete or absent. Historically every sector handled this inline (uniform 1/n,
country-mean, EU-mean, per-sector JSON overrides), which hides the provenance of the
values that end up weighting the proxy.

This module provides a single, explicit resolver:

- :class:`AlphaSource` enumerates the possible provenances.
- :func:`resolve_alpha` reads editable YAMLs under ``PROXY/config/ceip/alpha/fallback/`` and
  combines them with a caller-supplied ``reported`` value in a deterministic order:

    1. ``AlphaSource.REPORTED_COUNTRY`` - reported-emissions workbook had a value.
    2. ``AlphaSource.CONFIG_COUNTRY_OVERRIDE`` - a country-specific YAML entry.
    3. ``AlphaSource.CONFIG_DEFAULT`` - the cross-country defaults YAML.
    4. ``AlphaSource.UNIFORM_FALLBACK`` - ultimate fallback (``1/n`` across subsectors).

The resolver is additive: existing sector code paths keep working unchanged. Sectors opt
in by calling :func:`resolve_alpha` once they have attempted a reported-emissions read.

The YAML layout is:

.. code-block:: yaml

    # config/ceip/alpha/fallback/defaults.yaml
    version: 1
    sectors:
      B_Industry:
        G1:            # subsector group
          NH3: 0.25    # per-pollutant override
          CH4: 0.25
        G2:
          default: 0.25   # applies to any pollutant not listed above
      J_Waste:
        solid:
          default: 0.34

    # config/ceip/alpha/fallback/B_Industry_EL.yaml (per-country override)
    version: 1
    sector: B_Industry
    country: EL
    groups:
      G1:
        NH3: 0.30
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class AlphaSource(str, Enum):
    REPORTED_COUNTRY = "reported_country"
    CONFIG_COUNTRY_OVERRIDE = "config_country_override"
    CONFIG_DEFAULT = "config_default"
    UNIFORM_FALLBACK = "uniform_fallback"


@dataclass(frozen=True)
class AlphaResolution:
    """Result of :func:`resolve_alpha`.

    ``values`` is a mapping ``subsector_key -> alpha``; ``source`` is a mapping
    ``subsector_key -> AlphaSource`` so callers can log provenance per subsector.
    """

    values: dict[str, float]
    source: dict[str, "AlphaSource"] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def any_fallback(self) -> bool:
        return any(s is not AlphaSource.REPORTED_COUNTRY for s in self.source.values())


def _load_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001 - fallback must never crash the pipeline
        logger.warning("alpha.fallback: failed to read %s (%s); ignoring.", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _canonical_country(country: str) -> str:
    c = str(country).strip().upper()
    return {"GRC": "EL", "GR": "EL"}.get(c, c)


def _fallback_dir(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]
    primary = project_root / "PROXY" / "config" / "ceip" / "alpha" / "fallback"
    if primary.is_dir():
        return primary
    return project_root / "PROXY" / "config" / "alpha" / "fallback"


def _lookup(mapping: dict[str, Any], group: str, pollutant: str) -> float | None:
    if not isinstance(mapping, dict):
        return None
    grp = mapping.get(group) or mapping.get(str(group).upper()) or mapping.get(str(group).lower())
    if not isinstance(grp, dict):
        return None
    v = grp.get(pollutant)
    if v is None:
        v = grp.get("default")
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def resolve_alpha(
    *,
    sector: str,
    country: str,
    pollutant: str,
    subsectors: list[str],
    reported: dict[str, float] | None = None,
    project_root: Path | None = None,
) -> AlphaResolution:
    """Combine reported values with YAML fallbacks and a uniform last-resort split.

    Parameters
    ----------
    sector
        Sector key (e.g. ``"B_Industry"``).
    country
        Country code (EL / GR / GRC accepted; canonicalized to the short code).
    pollutant
        Pollutant label (passed through unchanged; YAML lookups are case-sensitive).
    subsectors
        Ordered list of subsector / group keys the caller expects to receive weights for.
    reported
        Optional mapping ``subsector -> value`` from the reported-emissions workbook.
        Missing entries (or non-finite) are treated as absent and filled from fallback.
    project_root
        Repo root. Auto-detected (4 levels up) when ``None``.

    Returns
    -------
    AlphaResolution
    """
    cc = _canonical_country(country)
    reported = dict(reported or {})

    defaults = _load_yaml_if_exists(_fallback_dir(project_root) / "defaults.yaml")
    country_path = _fallback_dir(project_root) / f"{sector}_{cc}.yaml"
    country_cfg = _load_yaml_if_exists(country_path)

    sector_defaults = ((defaults.get("sectors") or {}).get(sector) or {}) if isinstance(defaults, dict) else {}
    country_groups = country_cfg.get("groups") or {}

    values: dict[str, float] = {}
    source: dict[str, AlphaSource] = {}
    notes: list[str] = []

    for sub in subsectors:
        v = reported.get(sub)
        if v is not None:
            try:
                fv = float(v)
                if fv == fv:  # NaN check without importing math
                    values[sub] = fv
                    source[sub] = AlphaSource.REPORTED_COUNTRY
                    continue
            except (TypeError, ValueError):
                pass

        v = _lookup(country_groups, sub, pollutant)
        if v is not None:
            values[sub] = float(v)
            source[sub] = AlphaSource.CONFIG_COUNTRY_OVERRIDE
            notes.append(f"{sub}: country override ({country_path.name})")
            continue

        v = _lookup(sector_defaults, sub, pollutant)
        if v is not None:
            values[sub] = float(v)
            source[sub] = AlphaSource.CONFIG_DEFAULT
            notes.append(f"{sub}: sector default (defaults.yaml)")
            continue

    # Uniform fallback only fills the remainder (keeps reported/config values unchanged).
    missing = [s for s in subsectors if s not in values]
    if missing:
        n = max(len(subsectors), 1)
        uniform = 1.0 / n
        for s in missing:
            values[s] = uniform
            source[s] = AlphaSource.UNIFORM_FALLBACK
        notes.append(
            f"uniform 1/{n} fill for {len(missing)} subsector(s): {missing}"
        )

    return AlphaResolution(values=values, source=source, notes=notes)


def format_provenance(res: AlphaResolution) -> str:
    """One-line log summary useful for sector pipelines."""
    counts: dict[str, int] = {}
    for s in res.source.values():
        counts[s.value] = counts.get(s.value, 0) + 1
    parts = [f"{k}={v}" for k, v in counts.items()]
    return "alpha_sources[" + ",".join(parts) + "]"
