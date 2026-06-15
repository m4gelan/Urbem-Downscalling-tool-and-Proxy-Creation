from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

COLOR_W = "#4f7cff"
COLOR_A = "#f5a623"
POLLUTANT_COLORS = {"entropy": COLOR_W, "dominant": "#e05c5c", "mid": "#8b91a8"}
SECTOR_COLORS = {
    "A_PublicPower": "#4f7cff",
    "B_Industry": "#3ecf8e",
    "D_Fugitive": "#f5a623",
    "I_Offroad": "#e05c5c",
    "J_Waste": "#c084fc",
}
A1_TERM_COLORS = ["#4f7cff", "#3ecf8e", "#f5a623", "#e05c5c", "#c084fc", "#8b91a8"]


def apply_thesis_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 11,
    })


def _save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_stability_summary(
    rows: list[dict[str, Any]],
    out: Path,
    *,
    pollutant: str | None = None,
) -> None:
    secs = [r["sector"] for r in rows]
    x = np.arange(len(secs))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(
        x - width / 2,
        [r["tv_w_mean"] for r in rows],
        width,
        yerr=[
            [0.0] * len(rows),
            [max(0.0, r["tv_w_p90"] - r["tv_w_mean"]) for r in rows],
        ],
        capsize=3,
        label=r"expert weights $w$",
        color=COLOR_W,
    )
    ax.bar(
        x + width / 2,
        [r["tv_a_mean"] for r in rows],
        width,
        yerr=[
            [0.0] * len(rows),
            [max(0.0, r["tv_a_p90"] - r["tv_a_mean"]) for r in rows],
        ],
        capsize=3,
        label=r"reported split $\alpha$",
        color=COLOR_A,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(secs, rotation=20, ha="right")
    ax.set_ylabel(r"within-cell mass reallocated  TV(c)")
    title = r"Sensitivity to $\pm20\%$ perturbation: $w$ vs $\alpha$"
    if pollutant:
        title = f"{pollutant} — {title}"
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out)


def plot_headline_w_vs_alpha_sector(
    sector: str,
    stats: dict[str, float],
    out: Path,
    *,
    pollutant: str,
    pct: float = 0.2,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    labels = [r"expert $w$", r"split $\alpha$"]
    means = [stats["tv_w_mean"], stats["tv_a_mean"]]
    caps = [
        max(0.0, stats["tv_w_p90"] - stats["tv_w_mean"]),
        max(0.0, stats["tv_a_p90"] - stats["tv_a_mean"]),
    ]
    colors = [COLOR_W, COLOR_A]
    for i, (m, cap, col, lab) in enumerate(zip(means, caps, colors, labels)):
        ax.bar(
            i,
            m,
            width=0.55,
            yerr=[[0.0], [cap]],
            capsize=4,
            color=col,
            label=lab,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"within-cell mass reallocated  TV(c)")
    ax.set_title(
        f"{sector} — {pollutant}\n"
        rf"Sensitivity to $\pm{int(pct * 100)}\%$ perturbation: $w$ vs $\alpha$"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out)


def plot_a1_cross_sector(
    sector_a1: dict[str, dict[int, float]],
    out: Path,
    *,
    pollutant: str | None = None,
) -> None:
    sectors = sorted(sector_a1.keys())
    all_k = sorted({k for d in sector_a1.values() for k in d})
    if not all_k:
        return
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(sectors) + 1.2))
    y = np.arange(len(sectors))
    left = np.zeros(len(sectors))
    for i, k in enumerate(all_k):
        vals = [sector_a1[s].get(k, 0.0) * 100.0 for s in sectors]
        ax.barh(
            y,
            vals,
            left=left,
            color=A1_TERM_COLORS[i % len(A1_TERM_COLORS)],
            label=f"{k} active terms",
        )
        left = left + vals
    ax.set_yticks(y)
    ax.set_yticklabels(sectors)
    ax.set_xlabel("CAMS mass share (%)")
    title = "A1 — active mixture terms per cell (max over groups)"
    if pollutant:
        title = f"{pollutant} — {title}"
    ax.set_title(title)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    fig.tight_layout(rect=[0, 0, 0.78, 1])
    _save(fig, out)


def plot_a2_overlay(
    results: list[dict[str, Any]],
    out: Path,
    threshold: float = 0.7,
    *,
    pollutant: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for r in results:
        samples = r.get("a2_similarity_samples") or []
        weights = r.get("a2_mass_weights") or []
        if not samples:
            continue
        sk = r["sector_key"]
        ax.hist(
            samples,
            bins=25,
            weights=weights,
            histtype="step",
            linewidth=1.4,
            color=SECTOR_COLORS.get(sk, COLOR_W),
            label=sk,
            density=True,
        )
    ax.axvline(threshold, color="#8b91a8", linestyle="--", linewidth=1.2, label=f"threshold {threshold}")
    ax.set_xlabel("Mean pairwise cosine similarity (multi-term cells)")
    ax.set_ylabel("CAMS mass-weighted density")
    title = "A2 — term similarity across sectors (w)"
    if pollutant:
        title = f"{pollutant} — {title}"
    ax.set_title(title)
    ax.legend(
        frameon=False,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
    )
    fig.tight_layout(rect=[0, 0, 0.76, 1])
    _save(fig, out)


def plot_a3_sensitivity_scatter(
    points: list[dict[str, Any]],
    out: Path,
    *,
    pollutant: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for p in points:
        c = POLLUTANT_COLORS.get(p["pollutant_role"], COLOR_W)
        ax.scatter(
            p["a3_w"],
            p["tv_w_mean"],
            s=70,
            c=c,
            edgecolors="white",
            linewidths=0.4,
            label=p.get("legend_label"),
        )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
    ax.set_xlabel(r"w-exposure A3 (CAMS mass fraction)")
    ax.set_ylabel(r"mean TV(c) under $\pm20\%$ $w$ perturb")
    title = "Conditioning vs sensitivity (per sector)"
    if pollutant:
        title = f"{pollutant} — {title}"
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, out)


def plot_b2_violin_by_weight(
    weight_tv: list[tuple[str, list[float]]],
    out: Path,
    *,
    sector: str,
    pollutant: str | None = None,
) -> None:
    if not weight_tv:
        return
    labels = [t[0] for t in weight_tv]
    data = [t[1] for t in weight_tv]
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(labels)), 4))
    parts = ax.violinplot(data, positions=range(len(labels)), showmeans=True, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(COLOR_W)
        b.set_alpha(0.75)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("TV(c)")
    title = f"B2 — per-weight TV distribution ({sector}, ±20% $w$)"
    if pollutant:
        title = f"{pollutant} — {title}"
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, out)


def plot_b3(
    W0: np.ndarray,
    W1: np.ndarray,
    cell_shape: tuple[int, int],
    out: Path,
    *,
    title: str,
) -> None:
    a = W0.reshape(cell_shape)
    b = W1.reshape(cell_shape)
    vmax = float(max(a.max(), b.max())) or 1e-12
    d = b - a
    dmax = float(np.abs(d).max()) or 1e-12
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    im = None
    for k, (m, t) in enumerate([(a, r"baseline $W$"), (b, r"$w$ perturbed")]):
        im = axes[k].imshow(m, vmin=0, vmax=vmax, cmap="viridis")
        axes[k].set_title(t)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    imd = axes[2].imshow(d, cmap="RdBu_r", norm=TwoSlopeNorm(0, -dmax, dmax))
    axes[2].set_title(rf"$\Delta W$ (max $|{dmax:.1e}|$)")
    fig.colorbar(imd, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out)


def plot_b4(df_rows: list[tuple[str, str, str, float]], out: Path) -> None:
    rows = sorted(df_rows, key=lambda r: r[-1])
    labels = [f"{r[0]} · {r[1]} · {r[2]}" for r in rows]
    vals = [r[-1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(vals) + 1))
    ax.barh(range(len(vals)), vals, color=COLOR_W)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$\partial\,\mathrm{TV}/\partial w$ (±20% bump)")
    fig.tight_layout()
    _save(fig, out)
