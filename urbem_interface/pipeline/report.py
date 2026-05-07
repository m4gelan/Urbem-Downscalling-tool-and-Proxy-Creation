"""Text report for proxy pipeline (downscaling roles vs auxiliary factory outputs)."""

from __future__ import annotations

from pathlib import Path

from urbem_interface.pipeline.catalog import (
    ProxyPipelineBundle,
    build_proxy_pipeline_bundle,
    merge_proxy_catalog,
    validate_proxy_files,
)


def render_proxy_pipeline_report(
    bundle: ProxyPipelineBundle,
    *,
    proxies_folder: Path | None = None,
) -> str:
    lines: list[str] = []
    lines.append("=== UrbEm proxy pipeline report ===")
    lines.append("")
    lines.append("--- Downscaling (gnfr_to_proxy -> proxies key -> file) ---")
    for gnfr, role in sorted(bundle.gnfr_to_proxy.items()):
        fname = bundle.role_to_file.get(role, "?")
        sem = bundle.semantic_proxy_roles.get(role, "")
        extra = f"  [{sem}]" if sem else ""
        lines.append(f"  {gnfr} -> {role} -> {fname}{extra}")
    lines.append("")
    lines.append("--- Proxy role keys not assigned to any GNFR ---")
    orphan = set(bundle.role_to_file.keys()) - bundle.downscaling_roles
    if not orphan:
        lines.append("  (none)")
    else:
        for r in sorted(orphan):
            lines.append(f"  {r} -> {bundle.role_to_file[r]}")
    lines.append("")
    spm = bundle.raw.get("snap_proxy_map") or {}
    if spm:
        lines.append("--- snap_proxy_map (future SNAP-level downscaling) ---")
        for sk in sorted(spm.keys(), key=lambda x: (len(str(x)), str(x))):
            lines.append(f"  SNAP {sk} -> proxy id {spm[sk]!r}")
        lines.append("")

    lines.append("--- Auxiliary / factory-only (not in gnfr_to_proxy) ---")
    for a in bundle.auxiliary:
        snap = f" SNAP{a.snap}" if a.snap is not None else ""
        lines.append(f"  {a.id}{snap}  file={a.file}")
        if a.description:
            lines.append(f"      {a.description}")
    lines.append("")
    lines.append("--- Future disaggregation (reserved; not used by core yet) ---")
    fd = bundle.future
    for k in sorted(fd.keys()):
        if k.startswith("_"):
            lines.append(f"  {k}: {fd[k]}")
    agri = fd.get("agri_subproxies")
    if isinstance(agri, dict) and agri:
        lines.append("  agri_subproxies:")
        for rk, fn in agri.items():
            lines.append(f"    {rk} -> {fn}")
    else:
        lines.append("  agri_subproxies: {}  (add CORINE-split rasters + mapping for Topic 4)")
    pw = fd.get("pollutant_proxy_weights")
    if isinstance(pw, dict) and pw:
        lines.append("  pollutant_proxy_weights: (see JSON)")
    else:
        lines.append("  pollutant_proxy_weights: {}  (GNFR x pollutant -> proxy role)")
    lines.append("")
    if proxies_folder is not None:
        issues = validate_proxy_files(proxies_folder, bundle)
        lines.append("--- File check ---")
        if not issues:
            lines.append("  OK (all referenced paths exist under proxies_folder)")
        else:
            for msg in issues:
                lines.append(f"  MISSING: {msg}")
        lines.append("")
    return "\n".join(lines)


def report_from_proxies_json(
    proxies_json: Path,
    *,
    proxies_folder: Path | None = None,
) -> str:
    from urbem_interface.pipeline.job_config import materialize_proxies_config_file

    merged = materialize_proxies_config_file(
        Path(proxies_json), proxies_folder=proxies_folder
    )
    merged = merge_proxy_catalog(merged)
    bundle = build_proxy_pipeline_bundle(merged)
    return render_proxy_pipeline_report(bundle, proxies_folder=proxies_folder)
