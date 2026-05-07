"""Quick choropleth maps from pipeline outputs (config-driven, no CLI args)."""

from __future__ import annotations

import math
from typing import Any

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from ...config import project_root
from ...core import resolve_nuts_gpkg, resolve_path
from ...core.run_countries import parse_run_country_codes


def _to_long_weights(df: pd.DataFrame, pollutants: list[str]) -> pd.DataFrame:
    if {"pollutant", "NUTS_ID", "CLC_CODE", "w_p"}.issubset(df.columns):
        sub = df.copy()
        if pollutants:
            sub = sub[sub["pollutant"].isin(pollutants)].copy()
        return sub

    frames: list[pd.DataFrame] = []
    for pollutant in pollutants:
        col = f"W_{pollutant.replace('.', '_')}"
        if col not in df.columns:
            continue
        sub = df[["NUTS_ID", "CLC_CODE"]].copy()
        for opt in ("NAME_REGION", "COUNTRY"):
            if opt in df.columns:
                sub[opt] = df[opt]
        sub["pollutant"] = pollutant
        sub["w_p"] = df[col]
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["NUTS_ID", "CLC_CODE", "pollutant", "w_p"])
    return pd.concat(frames, ignore_index=True)


def run_visualization(cfg: dict[str, Any], combined_df: pd.DataFrame | None = None) -> None:
    vis = cfg.get("visualization") or {}
    root = project_root(cfg)
    paths = cfg.get("paths") or {}
    outputs = paths.get("outputs") or {}
    geom = paths.get("geometry") or {}
    run = cfg.get("run") or {}

    country_codes = parse_run_country_codes(run)
    clc_plot = int(vis.get("map_clc_code", 18))
    dpi = int(vis.get("dpi", 150))
    pollutants = [str(p) for p in (vis.get("plot_pollutants") or ["CH4", "NOx"])]
    out_dir = resolve_path(root, vis.get("output_dir", "Agriculture/results/plots"))

    combined_path = resolve_path(
        root, outputs.get("long_csv", outputs.get("combined_csv", "Agriculture/results/weights_long.csv"))
    )
    if combined_df is None:
        if not combined_path.is_file():
            print(f"Visualization skipped: {combined_path} not found.")
            return
        combined_df = pd.read_csv(combined_path)

    long_df = _to_long_weights(combined_df, pollutants)
    sub = long_df[long_df["CLC_CODE"] == clc_plot].copy()
    if sub.empty:
        print(f"Visualization: no rows for CLC_CODE={clc_plot}.")
        return

    gpkg = resolve_nuts_gpkg(root, geom.get("nuts_gpkg", "Data/geometry/NUTS_RG_20M_2021_3035.gpkg"))
    if not gpkg.is_file():
        print(f"Visualization skipped: NUTS gpkg not found: {gpkg}")
        return

    nuts = gpd.read_file(gpkg)
    nuts2 = nuts[nuts["LEVL_CODE"] == 2].copy()
    if country_codes:
        cc = nuts2["CNTR_CODE"].astype(str).str.strip().str.upper()
        nuts2 = nuts2[cc.isin(country_codes)].copy()

    nuts2["NUTS_ID"] = nuts2["NUTS_ID"].astype(str).str.strip()
    sub["NUTS_ID"] = sub["NUTS_ID"].astype(str).str.strip()

    g = nuts2.merge(sub, on="NUTS_ID", how="inner")
    if g.empty:
        print("Visualization: no overlap between NUTS and combined CSV.")
        return

    g = g.to_crs(4326)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(pollutants)
    ncols = min(3, max(1, n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 5.0 * nrows))
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    fig.suptitle(
        f"Agriculture proxy weights (CLC {clc_plot})"
        + (f" — {', '.join(sorted(country_codes))}" if country_codes else ""),
        fontsize=12,
    )

    palettes = ["Oranges", "Blues", "Greens", "Purples", "Reds", "Greys"]
    for idx, pollutant in enumerate(pollutants):
        ax = axes_list[idx]
        g_pol = g[g["pollutant"] == pollutant].copy()
        if g_pol.empty:
            ax.set_visible(False)
            continue
        vmin = float(g_pol["w_p"].min())
        vmax = float(g_pol["w_p"].max())
        if vmin == vmax:
            vmax = vmin + 1e-9
        cmap = palettes[idx % len(palettes)]
        g_pol.plot(
            column="w_p",
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolor="0.35",
            linewidth=0.35,
            legend=False,
        )
        ax.set_title(f"W_{pollutant}", fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_list[len(pollutants):]:
        ax.set_visible(False)

    plt.tight_layout()
    out_png = out_dir / f"ag_weights_clc{clc_plot}.png"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote map: {out_png}")
