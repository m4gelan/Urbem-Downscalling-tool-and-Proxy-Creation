from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from proxy.core import log
from proxy.diagnostics.weight_sensitivity.load_exports import bundle_path, load_bundle, mass_for_pollutant
from proxy.diagnostics.weight_sensitivity.mix_metrics import compute_prong_a_mix, precompute_mix_group_specs
from proxy.diagnostics.weight_sensitivity.prong_w_core import load_mix_from_manifest


def _plot_a1(a1: dict[int, float], out_path: Path, sector_key: str) -> None:
    keys = sorted(a1.keys())
    vals = [a1[k] * 100.0 for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(k) for k in keys], vals, color="#4f7cff")
    ax.set_xlabel("Active mixture terms per CAMS cell (max over groups)")
    ax.set_ylabel("CAMS mass share (%)")
    ax.set_title(f"{sector_key} — A1 active-term count (w)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_a2(samples: list[float], weights: list[float], out_path: Path, sector_key: str) -> None:
    if not samples:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=30, weights=weights, color="#4f7cff", alpha=0.85)
    ax.set_xlabel("Mean pairwise cosine similarity (multi-term cells)")
    ax.set_ylabel("CAMS mass-weighted count")
    ax.set_title(f"{sector_key} — A2 term similarity (w)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_a3_bar(rows: list[tuple[str, float]], out_path: Path) -> None:
    sectors = [r[0] for r in rows]
    fracs = [r[1] * 100.0 for r in rows]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(sectors))))
    ax.barh(sectors, fracs, color="#4f7cff")
    ax.set_xlabel("w-exposure mass fraction (%)")
    ax.set_title("A3 — w-sensitive mass (sectors with mix export)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_sector_prong_a_w(
    export_root: Path,
    sector_key: str,
    country_tag: str,
    year: int,
    *,
    active_eps: float,
    similarity_threshold: float,
    reference_pollutants: list[str],
    figures_dir: Path | None = None,
    write_per_sector_a1: bool = False,
) -> dict[str, Any]:
    bundle_dir = bundle_path(export_root, sector_key, country_tag, year)
    log.info(f"Prong A (w) {sector_key}: loading bundle {bundle_dir}")
    data = load_bundle(bundle_dir)
    if not data.get("has_mix"):
        raise ValueError(f"{sector_key}: bundle has no mix export — re-run build_and_export")
    mix_by_group = load_mix_from_manifest(data["manifest"], bundle_dir)
    log.info(f"Prong A (w) {sector_key}: mix specs loaded, aggregating cells")
    precomputed = precompute_mix_group_specs(mix_by_group, data["cell_id"], bundle_dir=bundle_dir)
    log.info(f"Prong A (w) {sector_key}: {len(precomputed[0])} footprint cells")
    by_pollutant: dict[str, Any] = {}
    for pol in reference_pollutants:
        mass = mass_for_pollutant(data["mass_by_pollutant"], pol)
        by_pollutant[pol] = compute_prong_a_mix(
            mix_by_group,
            data["cell_id"],
            mass,
            active_eps=active_eps,
            similarity_threshold=similarity_threshold,
            precomputed=precomputed,
        )
    result: dict[str, Any] = {
        "sector_key": sector_key,
        "country_tag": country_tag,
        "year": int(year),
        "bundle_dir": str(bundle_dir),
        "reference_pollutants": reference_pollutants,
        "by_pollutant": by_pollutant,
    }
    if reference_pollutants:
        result.update(by_pollutant[reference_pollutants[0]])

    out_dir = figures_dir or (
        export_root.parent / "figures" / "prong_a_w" / sector_key / f"{country_tag}_{year}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if write_per_sector_a1 and reference_pollutants:
        _plot_a1(
            by_pollutant[reference_pollutants[0]]["a1_mass_share_by_active_terms"],
            out_dir / "a1_stacked.png",
            sector_key,
        )
    log.info(f"Prong A (w) {sector_key}: {len(reference_pollutants)} pollutants → {out_dir}")
    return result


def run_all_prong_a_w(
    export_root: Path,
    country_tag: str,
    year: int,
    sector_keys: tuple[str, ...] | list[str],
    *,
    active_eps: float,
    similarity_threshold: float,
    reference_pollutants: list[str],
) -> list[dict[str, Any]]:
    rollup_dir = export_root.parent / "figures" / "prong_a_w"
    rollup_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    keys = list(sector_keys)
    log.info(f"Prong A (w): {len(keys)} sectors for {country_tag}_{year}")
    for i, sk in enumerate(keys, start=1):
        bdir = bundle_path(export_root, sk, country_tag, year)
        if not bdir.is_dir():
            log.warning(f"Prong A (w) skip {sk}: no bundle at {bdir}")
            continue
        log.info(f"Prong A (w) [{i}/{len(keys)}] {sk}")
        try:
            results.append(
                run_sector_prong_a_w(
                    export_root,
                    sk,
                    country_tag,
                    year,
                    active_eps=active_eps,
                    similarity_threshold=similarity_threshold,
                    reference_pollutants=reference_pollutants,
                )
            )
        except ValueError as exc:
            log.warning(f"Prong A (w) skip {sk}: {exc}")

    if results:
        import csv

        csv_path = rollup_dir / f"all_sectors_a3_{country_tag}_{year}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pollutant", "sector", "a3_w_exposure_fraction", "a2_mean_similarity", "n_mix_groups"])
            for pol in reference_pollutants:
                rows = [
                    (r["sector_key"], r["by_pollutant"][pol]["a3_w_exposure_fraction"])
                    for r in results
                ]
                _plot_a3_bar(rows, rollup_dir / f"a3_cross_sector_{country_tag}_{year}__{pol}.png")
                for r in results:
                    bp = r["by_pollutant"][pol]
                    w.writerow([
                        pol,
                        r["sector_key"],
                        bp["a3_w_exposure_fraction"],
                        bp["a2_mean_similarity_multi_term"],
                        bp["n_mix_groups"],
                    ])
        log.info(f"Prong A (w) rollup: {csv_path}")
    return results
