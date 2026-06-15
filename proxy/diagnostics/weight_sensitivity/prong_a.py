from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from proxy.core import log
from proxy.diagnostics.weight_sensitivity.cell_metrics import compute_prong_a
from proxy.diagnostics.weight_sensitivity.load_exports import (
    MULTI_GROUP_SECTORS,
    bundle_path,
    load_bundle,
    mass_for_pollutant,
)


def _plot_a1(a1: dict[int, float], out_path: Path, sector_key: str) -> None:
    keys = sorted(a1.keys())
    vals = [a1[k] * 100.0 for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(k) for k in keys], vals, color="#4f7cff")
    ax.set_xlabel("Active groups per CAMS cell")
    ax.set_ylabel("CAMS mass share (%)")
    ax.set_title(f"{sector_key} — A1 active-group count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_a2(samples: list[float], weights: list[float], out_path: Path, sector_key: str) -> None:
    if not samples:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=30, weights=weights, color="#4f7cff", alpha=0.85)
    ax.set_xlabel("Mean pairwise cosine similarity (multi-group cells)")
    ax.set_ylabel("CAMS mass-weighted count")
    ax.set_title(f"{sector_key} — A2 group similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_a3_bar(rows: list[tuple[str, float]], out_path: Path) -> None:
    sectors = [r[0] for r in rows]
    fracs = [r[1] * 100.0 for r in rows]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(sectors))))
    ax.barh(sectors, fracs, color="#4f7cff")
    ax.set_xlabel("α-exposure mass fraction (%)")
    ax.set_title("A3 — weight-sensitive mass (multi-group sectors)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_sector_prong_a(
    export_root: Path,
    sector_key: str,
    country_tag: str,
    year: int,
    *,
    active_eps: float,
    similarity_threshold: float,
    reference_pollutants: list[str],
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    bundle_dir = bundle_path(export_root, sector_key, country_tag, year)
    log.info(f"Prong A {sector_key}: loading bundle {bundle_dir}")
    data = load_bundle(bundle_dir)
    n_px = int(data["cell_id"].size)
    log.info(f"Prong A {sector_key}: grid {data['cell_id'].shape} ({n_px} px), {len(data['W_by_group'])} groups")
    by_pollutant: dict[str, Any] = {}
    for pol in reference_pollutants:
        mass = mass_for_pollutant(data["mass_by_pollutant"], pol)
        by_pollutant[pol] = compute_prong_a(
            data["W_by_group"],
            data["cell_id"],
            mass,
            active_eps=active_eps,
            similarity_threshold=similarity_threshold,
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
        primary = reference_pollutants[0]
        result.update(by_pollutant[primary])

    out_dir = figures_dir or (export_root.parent / "figures" / "prong_a" / sector_key / f"{country_tag}_{year}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info(f"Prong A {sector_key}: {len(reference_pollutants)} pollutants → {out_dir}")
    return result


def run_all_prong_a(
    export_root: Path,
    country_tag: str,
    year: int,
    sector_keys: tuple[str, ...] | list[str],
    *,
    active_eps: float,
    similarity_threshold: float,
    reference_pollutants: list[str],
) -> list[dict[str, Any]]:
    rollup_dir = export_root.parent / "figures" / "prong_a"
    rollup_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    keys = list(sector_keys)
    log.info(f"Prong A: {len(keys)} sectors for {country_tag}_{year}")
    for i, sk in enumerate(keys, start=1):
        bdir = bundle_path(export_root, sk, country_tag, year)
        if not bdir.is_dir():
            log.warning(f"Prong A skip {sk}: no bundle at {bdir}")
            continue
        log.info(f"Prong A [{i}/{len(keys)}] {sk}")
        results.append(
            run_sector_prong_a(
                export_root,
                sk,
                country_tag,
                year,
                active_eps=active_eps,
                similarity_threshold=similarity_threshold,
                reference_pollutants=reference_pollutants,
            )
        )

    if results:
        import csv

        csv_path = rollup_dir / f"all_sectors_a3_{country_tag}_{year}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pollutant", "sector", "a3_alpha_exposure_fraction", "a2_mean_similarity", "n_groups"])
            for pol in reference_pollutants:
                rows = [
                    (r["sector_key"], r["by_pollutant"][pol]["a3_alpha_exposure_fraction"])
                    for r in results
                ]
                _plot_a3_bar(rows, rollup_dir / f"a3_cross_sector_{country_tag}_{year}__{pol}.png")
                for r in results:
                    bp = r["by_pollutant"][pol]
                    w.writerow([
                        pol,
                        r["sector_key"],
                        bp["a3_alpha_exposure_fraction"],
                        bp["a2_mean_similarity_multi_group"],
                        bp["n_groups"],
                    ])
        log.info(f"Prong A rollup: {csv_path}")
    return results
