from __future__ import annotations
from pathlib import Path
import numpy as np
from proxy.core import log
from proxy.dataset_loaders.load_ef_c_othercombustion import (
    ef_kg_per_tj,
    load_emep,
    parse_emep_tables,
)
from proxy.dataset_loaders.load_c_othercombustion_specific import (
    EurostatEndUseResult,
    GainsActivityResult,
    compute_eurostat_f_enduse,
    compute_gains_activity,
    gains_dom_share_path,
    log_eurostat_end_use_debug,
    log_gains_activity_debug,
)

MODEL_CLASSES: tuple[str, ...] = (
    "R_FIREPLACE",
    "R_HEATING_STOVE",
    "R_COOKING_STOVE",
    "R_BOILER_MAN",
    "R_BOILER_AUT",
    "C_BOILER_MAN",
    "C_BOILER_AUT",
)

def assemble_m_matrix(
    gains: GainsActivityResult,
    enduse: EurostatEndUseResult,
    emep_rows: list,
    pollutant_outputs: list[str],
) -> np.ndarray:
    """
    M[p, k] = f_enduse(k) * f_GAINS(k) * sum_{r in class k} s_r * EF(p, class, fuel, appliance).
    EF from EMEP tables keyed by gains_class + gains_fuel (see EMEP_emission_factors.yaml).
    """
    pols = [str(p).strip() for p in pollutant_outputs if str(p).strip()]
    class_index = {c: i for i, c in enumerate(MODEL_CLASSES)}
    m = np.zeros((len(pols), len(MODEL_CLASSES)), dtype=np.float64)
    for row in gains.rows:
        ki = class_index.get(row.class_name)
        if ki is None:
            continue
        fe = float(enduse.f_enduse_by_class.get(row.class_name, 1.0))
        fg = float(gains.f_gains_by_class.get(row.class_name, 0.0))
        if fe <= 0.0 or fg <= 0.0 or row.s_r <= 0.0:
            continue
        scale = fe * fg * row.s_r
        for pi, pol in enumerate(pols):
            ef = ef_kg_per_tj(
                pol,
                row.class_name,
                row.fuel,
                row.appliance,
                emep_rows,
            )
            m[pi, ki] += scale * ef
    return m

def log_m_matrix(m: np.ndarray, pollutant_outputs: list[str]) -> None:
    pols = [str(p).strip() for p in pollutant_outputs if str(p).strip()]
    log.info("--- M matrix (f_enduse * f_GAINS * s_r * EF_g_per_GJ) ---")
    for line in _m_matrix_csv_lines(m, pols):
        log.info(line)


def _m_matrix_csv_lines(m: np.ndarray, pols: list[str]) -> list[str]:
    hdr = "pollutant," + ",".join(MODEL_CLASSES)
    rows = [hdr]
    for pi, pol in enumerate(pols):
        vals = ",".join(f"{m[pi, ci]:.6g}" for ci in range(len(MODEL_CLASSES)))
        rows.append(f"{pol},{vals}")
    return rows


def _dict_lines(title: str, data: dict[str, float], *, value_fmt: str = ".6g") -> list[str]:
    lines = [title]
    if not data:
        lines.append("  (none)")
        return lines
    w = max(len(k) for k in data)
    for k in sorted(data):
        lines.append(f"  {k:<{w}}  {data[k]:{value_fmt}}")
    return lines


def write_area_weights_debug(
    path: Path,
    gains: GainsActivityResult,
    enduse: EurostatEndUseResult,
    m: np.ndarray,
    pollutant_outputs: list[str],
) -> None:
    """Human-readable Eurostat / GAINS / M dump for area_weights_debug.txt."""
    pols = [str(p).strip() for p in pollutant_outputs if str(p).strip()]
    lines: list[str] = [
        "C_OtherCombustion — area weights debug (M = f_enduse * f_GAINS * s_r * EF)",
        "=" * 72,
        "",
        f"EUROSTAT  year={enduse.year}  iso3={enduse.iso3}  geo={enduse.geo}",
    ]
    if enduse.commercial_tj:
        cp = float(enduse.commercial_tj.get("FC_OTH_CP_E", 0.0))
        hh = float(enduse.commercial_tj.get("FC_OTH_HH_E", 0.0))
        tot = cp + hh
        lines += [
            "",
            "Commercial vs residential split (nrg_bal_s, TJ):",
            f"  FC_OTH_CP_E (commercial)     {cp:>12.3f}",
            f"  FC_OTH_HH_E (households)   {hh:>12.3f}",
            f"  total                      {tot:>12.3f}",
            f"  commercial_alpha (CP/total) {enduse.commercial_alpha:.6g}"
            if enduse.commercial_alpha is not None
            else "  commercial_alpha             (n/a)",
            f"  residential_share (HH/total) {enduse.residential_share:.6g}",
        ]
    lines += ["", *_dict_lines("Household end-use energy (TJ, nrg_d_hhq)", enduse.tj_by_metric, value_fmt=".3f")]
    if enduse.tj_by_metric:
        tj_sum = sum(enduse.tj_by_metric.values())
        lines.append(f"  (sum {tj_sum:.3f} TJ)")
    lines += [
        "",
        *_dict_lines("Residential metric shares f_by_metric", enduse.f_by_metric),
        "",
        *_dict_lines("f_enduse by bucket", enduse.f_enduse_by_bucket),
        "",
        "f_enduse by class (bucket in parentheses):",
    ]
    for cls in sorted(enduse.f_enduse_by_class):
        b = enduse.bucket_for_class.get(cls, "?")
        v = enduse.f_enduse_by_class[cls]
        lines.append(f"  {cls:<16}  {v:.6g}  ({b})")
    if enduse.api_sources:
        lines += ["", "Eurostat API sources:", *[f"  {k}: {v}" for k, v in sorted(enduse.api_sources.items())]]

    lines += [
        "",
        "=" * 72,
        "",
        f"GAINS  year={gains.year}  country={gains.country_tag!r}  "
        f"mapped_rows={len(gains.rows)}  unmapped_rows={len(gains.unmapped_rows)}",
        "",
        *_dict_lines("f_GAINS by class", gains.f_gains_by_class),
        "",
        *_dict_lines("Activity weight w_sum by class", gains.w_sum_by_class, value_fmt=".4f"),
        "",
        *_dict_lines("Activity weight w_sum by bucket", gains.w_sum_by_bucket, value_fmt=".4f"),
        "",
        "Mapped GAINS rows (fuel | appliance -> class, bucket, share, s_r):",
        f"  {'fuel':<36} {'appliance':<48} {'class':<16} {'bucket':<22} {'share':>7} {'s_r':>9}",
        "  " + "-" * 140,
    ]
    for r in sorted(gains.rows, key=lambda x: (x.class_name, -x.s_r, x.fuel, x.appliance)):
        fuel = r.fuel if len(r.fuel) <= 36 else r.fuel[:33] + "..."
        app = r.appliance if len(r.appliance) <= 48 else r.appliance[:45] + "..."
        lines.append(
            f"  {fuel:<36} {app:<48} {r.class_name:<16} {r.bucket:<22} "
            f"{r.share:>7.4f} {r.s_r:>9.6g}"
        )
    if gains.unmapped_rows:
        lines += ["", "Unmapped GAINS rows (fuel | appliance | share):"]
        for fuel, app, share in gains.unmapped_rows[:30]:
            lines.append(f"  {share:.6g}  |  {fuel}  |  {app}")
        if len(gains.unmapped_rows) > 30:
            lines.append(f"  ... and {len(gains.unmapped_rows) - 30} more")

    lines += [
        "",
        "=" * 72,
        "",
        "M matrix  [rows=pollutant, cols=class]  kg/TJ after f_enduse * f_GAINS * s_r * EF",
        "",
        *_m_matrix_csv_lines(m, pols),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def build_gains_activity_for_country(
    gains_folder: Path,
    gains_mapping_filepath: Path,
    year_gains: int,
    country_profile: dict[str, str],
) -> GainsActivityResult:
    gains_path = gains_dom_share_path(gains_folder, country_profile)
    mapping_path = Path(gains_mapping_filepath)
    log.info(f"C_OtherCombustion GAINS file: {gains_path.name}")
    result = compute_gains_activity(
        gains_path,
        int(year_gains),
        mapping_path,
        country_profile=country_profile,
    )
    log_gains_activity_debug(result)
    return result

def build_m_matrix(
    repo_root: Path,
    gains_folder: Path,
    gains_mapping_filepath: Path,
    eurostat_end_use_config: Path,
    emep_yaml_filepath: Path,
    year_gains: int,
    country_profile: dict[str, str],
    pollutant_outputs: list[str],
    *,
    eurostat_enabled: bool = True,
) -> tuple[GainsActivityResult, EurostatEndUseResult, np.ndarray]:
    gains = build_gains_activity_for_country(
        Path(gains_folder),
        Path(gains_mapping_filepath),
        year_gains,
        country_profile,
    )
    enduse = compute_eurostat_f_enduse(
        Path(repo_root),
        country_profile,
        Path(eurostat_end_use_config),
        enabled=eurostat_enabled,
    )
    log_eurostat_end_use_debug(enduse, country_profile)
    emep = load_emep(Path(emep_yaml_filepath))
    emep_rows = parse_emep_tables(emep)
    log.info(f"C_OtherCombustion EMEP: {len(emep_rows)} rows with gains_class + gains_fuel mapping")
    m = assemble_m_matrix(gains, enduse, emep_rows, pollutant_outputs)
    log_m_matrix(m, pollutant_outputs)
    return gains, enduse, m
