"""Lightweight alpha vector helper for optional ``reported`` dicts.

Country YAML trees under ``config/ceip/alpha/fallback/`` were removed; sector behaviour is
driven by ``alpha_methods.yaml`` and CEIP workbooks. This module keeps :func:`resolve_alpha`
as a small API: fill from ``reported`` where finite, else uniform ``1/n`` across ``subsectors``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AlphaSource(str, Enum):
    REPORTED_COUNTRY = "reported_country"
    CONFIG_COUNTRY_OVERRIDE = "config_country_override"
    CONFIG_DEFAULT = "config_default"
    UNIFORM_FALLBACK = "uniform_fallback"


@dataclass(frozen=True)
class AlphaResolution:
    """Result of :func:`resolve_alpha`."""

    values: dict[str, float]
    source: dict[str, "AlphaSource"] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def any_fallback(self) -> bool:
        return any(s is not AlphaSource.REPORTED_COUNTRY for s in self.source.values())


def resolve_alpha(
    *,
    sector: str,
    country: str,
    pollutant: str,
    subsectors: list[str],
    reported: dict[str, float] | None = None,
    project_root: Path | None = None,
) -> AlphaResolution:
    """Use ``reported`` entries where finite; otherwise uniform ``1/n`` per missing key."""
    _ = sector, country, pollutant, project_root
    reported = dict(reported or {})
    values: dict[str, float] = {}
    source: dict[str, AlphaSource] = {}
    notes: list[str] = []
    n = max(len(subsectors), 1)
    uniform = 1.0 / n

    for sub in subsectors:
        v = reported.get(sub)
        ok = False
        if v is not None:
            try:
                fv = float(v)
                if fv == fv:
                    values[sub] = fv
                    source[sub] = AlphaSource.REPORTED_COUNTRY
                    ok = True
            except (TypeError, ValueError):
                pass
        if ok:
            continue
        values[sub] = uniform
        source[sub] = AlphaSource.UNIFORM_FALLBACK

    if any(s is AlphaSource.UNIFORM_FALLBACK for s in source.values()):
        notes.append(f"uniform 1/{n} where reported missing ({sector})")
    return AlphaResolution(values=values, source=source, notes=notes)


def format_provenance(res: AlphaResolution) -> str:
    """One-line log summary."""
    counts: dict[str, int] = {}
    for s in res.source.values():
        counts[s.value] = counts.get(s.value, 0) + 1
    parts = [f"{k}={v}" for k, v in counts.items()]
    return "alpha_sources[" + ",".join(parts) + "]"
