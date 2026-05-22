from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ClcRateCell:
    nuts2: str
    l3: int
    rate: float
    n_lucas: int
    source: str
    eu_mean: float


@dataclass
class Lc1RateSignalDebug:
    title: str
    units: str
    l3_codes: list[int]
    lc1_ef: dict[str, float]
    cells: list[ClcRateCell] = field(default_factory=list)
    n_lucas_points: int = 0


@dataclass
class ManureNutsDebug:
    nuts2: str
    n_fodder: float
    n_nonfodder: float
    n_total: float
    r_fodder: float
    r_high: float
    r_med: float
    r_low: float
    ha_fodder: float
    ha_high: float
    ha_med: float
    ha_low: float


@dataclass
class KAgAreaWeightsDebug:
    country_iso3: str
    country_other: str
    lambda_h: dict[str, float] = field(default_factory=dict)
    manure: list[ManureNutsDebug] = field(default_factory=list)
    lc1_signals: list[Lc1RateSignalDebug] = field(default_factory=list)


def _matrix_from_cells(
    cells: list[ClcRateCell],
    l3_codes: list[int],
    nuts_order: list[str],
) -> dict[str, dict[int, ClcRateCell]]:
    by: dict[str, dict[int, ClcRateCell]] = {}
    for c in cells:
        by.setdefault(c.nuts2, {})[c.l3] = c
    return by


def _format_lc1_ef(lc1_ef: dict[str, float]) -> list[str]:
    lines = ["LC1 factors:"]
    if not lc1_ef:
        lines.append("  (none)")
        return lines
    w = max(len(k) for k in lc1_ef)
    for k in sorted(lc1_ef):
        lines.append(f"  {k:<{w}}  {lc1_ef[k]:.6g}")
    return lines


def _format_nuts_l3_table(sig: Lc1RateSignalDebug, nuts_order: list[str]) -> list[str]:
    l3s = sig.l3_codes
    if not l3s:
        return ["  (no CORINE L3 classes)"]
    by = _matrix_from_cells(sig.cells, l3s, nuts_order)
    hdr = "NUTS2  " + "".join(f"{c:>12}" for c in l3s)
    lines = [
        f"LUCAS points used: {sig.n_lucas_points}",
        f"CORINE L3 classes: {', '.join(str(c) for c in l3s)}",
        "",
        "Fallback rate (value used when LUCAS buffer is empty):",
        hdr,
    ]
    for nid in nuts_order:
        if nid not in by:
            continue
        row = f"{nid:<7}"
        for l3 in l3s:
            cell = by.get(nid, {}).get(l3)
            row += f"{(cell.rate if cell else 0.0):>12.6g}"
        lines.append(row)
    lines += [
        "",
        "Fallback provenance (n_LUCAS | source):",
        hdr.replace("NUTS2", "NUTS2  "),
    ]
    for nid in nuts_order:
        if nid not in by:
            continue
        row = f"{nid:<7}"
        for l3 in l3s:
            cell = by.get(nid, {}).get(l3)
            if cell is None:
                row += f"{'—':>12}"
            else:
                row += f"{cell.n_lucas:>4}/{cell.source:<7}"[:12].rjust(12)
        lines.append(row)
    lines.append("  source: nuts2=regional LUCAS mean, country=country-wide, eu_mean=EF table mean")
    return lines


def write_k_agriculture_area_weights_debug(path: Path, dump: KAgAreaWeightsDebug) -> None:
    nuts_order = sorted(
        set(dump.lambda_h)
        | {m.nuts2 for m in dump.manure}
        | {c.nuts2 for s in dump.lc1_signals for c in s.cells}
    )
    lines: list[str] = [
        "K_Agriculture — area weights debug",
        "=" * 72,
        f"country ISO3={dump.country_iso3}  LUCAS filter={dump.country_other}",
        "",
        "=" * 72,
        "Livestock housing — lambda_H by NUTS2",
        "",
    ]
    if dump.lambda_h:
        w = max(len(k) for k in dump.lambda_h)
        for k in sorted(dump.lambda_h):
            lines.append(f"  {k:<{w}}  {dump.lambda_h[k]:.6g}")
    else:
        lines.append("  (none)")

    lines += ["", "=" * 72, "Manure application — N pools and R (kg N ha-1 yr-1) by crop group", ""]
    if dump.manure:
        lines.append(
            f"  {'NUTS2':<7} {'N_fod':>10} {'N_non':>10} {'R_fod':>8} {'R_hi':>8} "
            f"{'R_med':>8} {'R_lo':>8}  ha_fod/ha_hi/ha_med/ha_lo"
        )
        for m in sorted(dump.manure, key=lambda x: x.nuts2):
            lines.append(
                f"  {m.nuts2:<7} {m.n_fodder:>10.4g} {m.n_nonfodder:>10.4g} "
                f"{m.r_fodder:>8.4g} {m.r_high:>8.4g} {m.r_med:>8.4g} {m.r_low:>8.4g}  "
                f"{m.ha_fodder:.3g}/{m.ha_high:.3g}/{m.ha_med:.3g}/{m.ha_low:.3g}"
            )
    else:
        lines.append("  (none)")

    for sig in dump.lc1_signals:
        lines += [
            "",
            "=" * 72,
            f"{sig.title}",
            f"Units: {sig.units}",
            "",
            *_format_lc1_ef(sig.lc1_ef),
            "",
            *_format_nuts_l3_table(sig, nuts_order),
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
