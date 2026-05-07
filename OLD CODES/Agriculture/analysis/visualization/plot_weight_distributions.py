"""Plots for selected-region baseline vs upgraded weight comparisons."""

from __future__ import annotations

from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLC_CLASS_NAMES = {
    12: "Non-irrigated arable land (211)",
    13: "Permanently irrigated land (212)",
    14: "Rice fields (213)",
    15: "Vineyards (221)",
    16: "Fruit trees and berry plantations (222)",
    17: "Olive groves (223)",
    18: "Pastures (231)",
    19: "Annual crops associated with permanent crops (241)",
    20: "Complex cultivation patterns (242)",
    21: "Land principally under agriculture with significant areas of natural vegetation (243)",
    22: "Agro-forestry areas (244)",
}

# Explicit national colors based on flags or widely used national palettes.
COUNTRY_COLORS = {
    "AT": "#ED2939",
    "BE": "#FCDD09",
    "BG": "#00966E",
    "CY": "#D57800",
    "CZ": "#11457E",
    "DE": "#000000",
    "DK": "#C60C30",
    "EE": "#4891D9",
    "EL": "#0D5EAF",
    "ES": "#AA151B",
    "FI": "#003580",
    "FR": "#0055A4",
    "HR": "#171796",
    "HU": "#436F4D",
    "IE": "#169B62",
    "IT": "#008C45",
    "LT": "#FDB913",
    "LU": "#00A3E0",
    "LV": "#9E3039",
    "MT": "#CF142B",
    "NL": "#FF4F00",
    "PL": "#DC143C",
    "PT": "#046A38",
    "RO": "#002B7F",
    "SE": "#006AA7",
    "SI": "#0056A3",
    "SK": "#0B4EA2",
}
FALLBACK_COUNTRY_COLOR = "#4C566A"


def _prep_plot_df(comparison_df: pd.DataFrame, pollutants: list[str]) -> pd.DataFrame:
    plot_df = comparison_df[comparison_df["pollutant"].isin(pollutants)].copy()
    plot_df["CLC_LABEL"] = plot_df["CLC_CODE"].astype(int).astype(str)
    return plot_df


def _region_file_token(country: str, nuts_id: str, name_region: str) -> str:
    raw = f"{country}_{nuts_id}_{name_region}"
    safe = "".join(ch if ch.isalnum() else "_" for ch in raw)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def plot_region_redistribution_heatmaps(
    redistribution_df: pd.DataFrame,
    pollutants: list[str],
    output_dir: Path,
    *,
    min_share: float = 0.025,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = redistribution_df[redistribution_df["pollutant"].isin(pollutants)].copy()
    if plot_df.empty:
        return

    plot_df = plot_df[plot_df["class_share"] >= float(min_share)].copy()
    if plot_df.empty:
        return

    region_meta = (
        plot_df[["COUNTRY", "NUTS_ID", "NAME_REGION"]]
        .drop_duplicates()
        .sort_values(["COUNTRY", "NAME_REGION", "NUTS_ID"])
    )
    for _, region in region_meta.iterrows():
        sub = plot_df[
            (plot_df["COUNTRY"] == region["COUNTRY"])
            & (plot_df["NUTS_ID"] == region["NUTS_ID"])
        ].copy()
        if sub.empty:
            continue

        class_order_df = (
            sub[["CLC_CODE", "class_share"]]
            .drop_duplicates()
            .sort_values(["class_share", "CLC_CODE"], ascending=[False, True])
        )
        ordered_classes = class_order_df["CLC_CODE"].astype(int).tolist()
        active_pollutants = [pollutant for pollutant in pollutants if pollutant in set(sub["pollutant"])]
        if not ordered_classes or not active_pollutants:
            continue

        pivot = (
            sub.pivot_table(
                index="CLC_CODE",
                columns="pollutant",
                values="delta",
                aggfunc="first",
            )
            .reindex(index=ordered_classes, columns=active_pollutants)
            .fillna(0.0)
        )
        if pivot.empty:
            continue

        max_abs = float(np.nanmax(np.abs(pivot.to_numpy()))) if pivot.size else 0.0
        vmax = max(max_abs, 1e-9)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        y_labels = []
        share_lookup = {
            int(row["CLC_CODE"]): float(row["class_share"])
            for _, row in class_order_df.iterrows()
        }
        for clc_code in ordered_classes:
            clc_name = CLC_CLASS_NAMES.get(int(clc_code), "Agricultural land cover")
            share_pct = 100.0 * share_lookup.get(int(clc_code), 0.0)
            y_labels.append(f"{clc_code} | {clc_name} ({share_pct:.1f}%)")

        fig_h = 0.55 * max(1, len(ordered_classes)) + 2.6
        fig, ax = plt.subplots(figsize=(8.8, fig_h))
        im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(active_pollutants)))
        ax.set_xticklabels(active_pollutants)
        ax.set_yticks(range(len(ordered_classes)))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("pollutant")
        ax.set_ylabel("CLC class")
        ax.set_title(
            f"{region['COUNTRY']} | {region['NAME_REGION']} ({region['NUTS_ID']})\n"
            f"Redistribution delta by pollutant and CLC class (share >= {100 * min_share:.1f}%)"
        )

        for row_idx in range(len(ordered_classes) + 1):
            ax.axhline(row_idx - 0.5, color="white", linewidth=0.8, alpha=0.8)
        for col_idx in range(len(active_pollutants) + 1):
            ax.axvline(col_idx - 0.5, color="white", linewidth=0.8, alpha=0.8)

        cbar = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
        cbar.set_label("delta = upgraded - baseline")

        fig.tight_layout()
        token = _region_file_token(
            str(region["COUNTRY"]),
            str(region["NUTS_ID"]),
            str(region["NAME_REGION"]),
        )
        fig.savefig(output_dir / f"{token}_redistribution_heatmap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_clc_pollutant_dumbbells(
    comparison_df: pd.DataFrame,
    pollutants: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = _prep_plot_df(comparison_df, pollutants)
    if plot_df.empty:
        return

    marker_handles = [
        Line2D(
            [0],
            [0],
            color="#666666",
            lw=0,
            marker="o",
            markersize=6,
            markerfacecolor="#666666",
            label="baseline",
        ),
        Line2D(
            [0],
            [0],
            color="#666666",
            lw=0,
            marker="D",
            markersize=6,
            markerfacecolor="#666666",
            label="upgraded",
        ),
    ]

    clc_codes = (
        plot_df["selected_clc_code"]
        if "selected_clc_code" in plot_df.columns
        else plot_df["CLC_CODE"]
    )
    for clc_code in sorted(pd.Series(clc_codes).dropna().astype(int).unique()):
        sub = plot_df[plot_df["CLC_CODE"].astype(int) == int(clc_code)].copy()
        if sub.empty:
            continue
        country_df = (
            sub[["COUNTRY"]]
            .drop_duplicates()
            .sort_values("COUNTRY")
            .reset_index(drop=True)
        )
        country_color_lookup = {
            row["COUNTRY"]: COUNTRY_COLORS.get(str(row["COUNTRY"]).upper(), FALLBACK_COUNTRY_COLOR)
            for _, row in country_df.iterrows()
        }
        region_names = (
            sub[["NUTS_ID", "NAME_REGION", "COUNTRY"]]
            .drop_duplicates()
            .sort_values(["COUNTRY", "NAME_REGION", "NUTS_ID"])
        )
        color_lookup = {
            row["NUTS_ID"]: country_color_lookup[row["COUNTRY"]]
            for _, row in region_names.iterrows()
        }
        region_handles = [
            Line2D(
                [0],
                [0],
                color=color_lookup[row["NUTS_ID"]],
                lw=2,
                marker="o",
                markersize=5,
                label=f"{row['COUNTRY']} | {row['NAME_REGION']}",
            )
            for _, row in region_names.iterrows()
        ]
        country_handles = [
            Line2D(
                [0],
                [0],
                color=country_color_lookup[row["COUNTRY"]],
                lw=2.5,
                label=row["COUNTRY"],
            )
            for _, row in country_df.iterrows()
        ]

        active_pollutants = [pollutant for pollutant in pollutants if pollutant in set(sub["pollutant"])]
        pollutant_to_y = {pollutant: idx for idx, pollutant in enumerate(active_pollutants)}
        offsets = np.linspace(-0.27, 0.27, num=max(1, len(region_names)))
        offset_lookup = {
            row["NUTS_ID"]: offsets[idx]
            for idx, (_, row) in enumerate(region_names.iterrows())
        }

        fig, ax = plt.subplots(figsize=(11, 0.85 * max(1, len(active_pollutants)) + 3.2))
        for idx in range(len(active_pollutants) - 1):
            ax.axhline(
                idx + 0.5,
                color="#B8BCC2",
                linewidth=0.8,
                linestyle=(0, (6, 4)),
                zorder=0,
            )
        for _, row in sub.iterrows():
            pollutant = str(row["pollutant"])
            if pollutant not in pollutant_to_y:
                continue
            y = pollutant_to_y[pollutant] + offset_lookup.get(row["NUTS_ID"], 0.0)
            color = color_lookup[row["NUTS_ID"]]
            ax.hlines(y, row["baseline_w"], row["w_p"], color=color, linewidth=2.0, alpha=0.9, zorder=1)
            ax.scatter(
                row["baseline_w"],
                y,
                color=color,
                s=36,
                marker="o",
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
            ax.scatter(
                row["w_p"],
                y,
                color=color,
                s=44,
                marker="D",
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

        ax.set_yticks(range(len(active_pollutants)))
        ax.set_yticklabels(active_pollutants)
        ax.set_xlabel("weight")
        ax.set_ylabel("pollutant")
        clc_name = CLC_CLASS_NAMES.get(int(clc_code), "Agricultural land cover")
        ax.set_title(f"CLC {clc_code} - {clc_name}\nBaseline vs upgraded pollutant weights")
        ax.grid(axis="x", color="#e0e0e0", linewidth=0.7)
        marker_legend = ax.legend(handles=marker_handles, title="Markers", loc="upper right", frameon=False)
        ax.add_artist(marker_legend)
        country_legend = ax.legend(handles=country_handles, title="Countries", loc="upper left", frameon=False)
        ax.add_artist(country_legend)
        fig.legend(
            handles=region_handles,
            title="Regions",
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(2, max(1, len(region_handles))),
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        fig.savefig(output_dir / f"CLC_{clc_code}_pollutant_dumbbell.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
