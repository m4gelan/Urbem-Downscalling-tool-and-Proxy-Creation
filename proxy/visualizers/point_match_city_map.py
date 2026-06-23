"""City point-source matching map from Proxy_weights *_matches.json.

Visualizes CAMS grid cells ↔ real facilities (RI-URBANS primary, JRC / E-PRTR /
OSM / CORINE / UWWTD fallback) within a city bounding box.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import contextily as cx
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pyproj import Transformer

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from UrbEm_Visualizer.visualization.map_config import sector_viz_meta

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BBOX_YAML = Path(__file__).resolve().parent / "bouding_boxes.yaml"

DEFAULT_SECTORS = [
    "A_PublicPower",
    "B_Industry",
    "D_Fugitive",
    "E_Solvents",
    "H_Aviation",
    "J_Waste",
]

ICON_GLYPH = {
    "bolt": "⚡", "factory": "🏭", "wind": "💨", "plane": "✈",
    "recycle": "♻", "flame": "🔥", "droplet": "💧", "ship": "⚓",
    "truck": "🚜", "leaf": "🌿", "dot": "●",
}

# Matplotlib markers (DejaVu-safe) for facility / sidebar icons
SECTOR_MARKERS = {
    "A_PublicPower": "*",
    "B_Industry": "D",
    "D_Fugitive": "D",
    "E_Solvents": "o",
    "H_Aviation": "^",
    "J_Waste": "h",
}

# Industry stands out on the map (distinct from other sector markers/colors)
FACILITY_STYLE: dict[str, dict[str, str]] = {
    "B_Industry": {"marker": "D", "color": "#c62828"},
}

SECTOR_SHORT = {
    "A_PublicPower": "Power",
    "B_Industry":    "Industry",
    "C_OtherStat":   "OtherStat",
    "D_Fugitive":    "Fugitive",
    "E_Solvents":    "Solvents",
    "F_RoadTransp":  "Road",
    "G_Shipping":    "Shipping",
    "H_Aviation":    "Aviation",
    "I_OffRoad":     "OffRoad",
    "J_Waste":       "Waste",
    "K_AgriLivestock": "Agri-Live",
    "L_AgriOther":   "Agri-Other",
}

DATASET_LABEL = {
    "riurbans": "RI-URBANS",
    "jrc": "JRC", "eprtr": "E-PRTR", "osm": "OSM",
    "corine": "CORINE", "uwwtd": "UWWTD",
}

DATASET_COLOR = {
    "riurbans": "#6a1b9a",
    "jrc":    "#546e7a",
    "eprtr":  "#0277bd",
    "osm":    "#3949ab",
    "corine": "#5d4037",
    "uwwtd":  "#00695c",
}

# CAMS grid resolution (degrees)
CAMS_DLON = 0.10
CAMS_DLAT = 0.05

_TO_M = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Match-status colors
COL_MATCHED   = "#2ca25f"   # green
COL_UNMATCHED = "#e34a33"   # red
COL_LINK      = "#2ca25f"
COL_BBOX      = "#1f77b4"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_bbox(city: str) -> list[float]:
    with open(BBOX_YAML, encoding="utf-8") as f:
        boxes = yaml.safe_load(f)["bounding_boxes"]
    if city not in boxes:
        raise KeyError(f"City {city!r} not in {BBOX_YAML}")
    return [float(x) for x in boxes[city]]


def _in_bbox(lon: float, lat: float, bbox: list[float]) -> bool:
    w, s, e, n = bbox
    return w <= lon <= e and s <= lat <= n


def _matches_path(proxy_root: Path, sector_id: str, country: str, year: int) -> Path:
    stem = f"{sector_id}_{country}_point_source_{year}"
    return proxy_root / sector_id / f"{stem}_matches.json"


def _match_dataset_key(row: dict[str, Any]) -> str:
    return str(row.get("dataset_key") or row.get("match_source") or "")


def _sidecar_facility_links(row: dict[str, Any]) -> list[dict[str, Any]]:
    links = row.get("facility_links")
    if isinstance(links, list) and links:
        return links
    if str(row.get("matched") or "no") != "yes":
        return []
    flon, flat = row.get("facility_lon"), row.get("facility_lat")
    if flon is not None and flat is not None:
        return [{
            "facility_id": row.get("facility_id"),
            "facility_lon": flon,
            "facility_lat": flat,
            "match_distance_km": row.get("match_distance_km"),
        }]
    return []


def _cams_summary(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for p in points:
        k = (p["sector_id"], p["cams_id"])
        if k not in by_key:
            by_key[k] = {
                "sector_id": p["sector_id"],
                "cams_id": p["cams_id"],
                "cams_lon": p["cams_lon"],
                "cams_lat": p["cams_lat"],
                "matched": False,
                "dataset_key": "",
            }
        if p["matched"]:
            by_key[k]["matched"] = True
            by_key[k]["dataset_key"] = p["dataset_key"]
    return list(by_key.values())


def _load_sector_points(
    proxy_root: Path,
    sectors: list[str],
    country: str,
    year: int,
    bbox: list[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sector_id in sectors:
        path = _matches_path(proxy_root, sector_id, country, year)
        if not path.is_file():
            print(f"  · skip {sector_id}: missing {path.name}")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for cams_id, row in rows.items():
            cams_lon = float(row["cams_lon"])
            cams_lat = float(row["cams_lat"])
            matched = str(row.get("matched") or "no") == "yes"
            ds_key = _match_dataset_key(row)
            cams_in = _in_bbox(cams_lon, cams_lat, bbox)
            link_rows = _sidecar_facility_links(row)

            if not link_rows:
                if not cams_in:
                    continue
                out.append({
                    "cams_id": str(cams_id),
                    "sector_id": sector_id,
                    "cams_lon": cams_lon,
                    "cams_lat": cams_lat,
                    "matched": matched,
                    "facility_lon": None,
                    "facility_lat": None,
                    "facility_id": None,
                    "dataset_key": ds_key,
                    "dataset": str(row.get("dataset") or ""),
                })
                continue

            for lk in link_rows:
                flon, flat = lk.get("facility_lon"), lk.get("facility_lat")
                fac_in = (
                    flon is not None
                    and flat is not None
                    and _in_bbox(float(flon), float(flat), bbox)
                )
                if not (cams_in or fac_in):
                    continue
                out.append({
                    "cams_id": str(cams_id),
                    "sector_id": sector_id,
                    "cams_lon": cams_lon,
                    "cams_lat": cams_lat,
                    "matched": matched,
                    "facility_lon": float(flon) if flon is not None else None,
                    "facility_lat": float(flat) if flat is not None else None,
                    "facility_id": lk.get("facility_id"),
                    "dataset_key": ds_key,
                    "dataset": str(row.get("dataset") or ""),
                })
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _xy(lon: float, lat: float) -> tuple[float, float]:
    return _TO_M.transform(lon, lat)


def _dataset_tag(rec: dict[str, Any]) -> str:
    key = rec["dataset_key"]
    if key in DATASET_LABEL:
        return DATASET_LABEL[key]
    if rec.get("dataset"):
        return str(rec["dataset"])
    return key.upper() if key else "?"


def _sector_color(sector_id: str) -> str:
    meta = sector_viz_meta(sector_id)
    return meta.get("accent", "#4f7cff")


def _sector_glyph(sector_id: str) -> str:
    meta = sector_viz_meta(sector_id)
    return ICON_GLYPH.get(meta.get("icon", "dot"), ICON_GLYPH["dot"])


def _sector_marker(sector_id: str) -> str:
    return SECTOR_MARKERS.get(sector_id, "o")


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_cams_cell(ax, rec: dict[str, Any]) -> None:
    """Draw the CAMS grid cell footprint as a small square (not a point).

    Conveys that CAMS is a *cell*, not a point source.
    """
    lon, lat = rec["cams_lon"], rec["cams_lat"]
    # Cell corners (cell-centered)
    w = lon - CAMS_DLON / 2
    e = lon + CAMS_DLON / 2
    s = lat - CAMS_DLAT / 2
    n = lat + CAMS_DLAT / 2
    (x0, y0) = _xy(w, s)
    (x1, y1) = _xy(e, n)
    color = COL_MATCHED if rec["matched"] else COL_UNMATCHED
    rect = Rectangle(
        (min(x0, x1), min(y0, y1)),
        abs(x1 - x0), abs(y1 - y0),
        facecolor=color, alpha=0.18,
        edgecolor=color, linewidth=0.8,
        linestyle="-" if rec["matched"] else "--",
        zorder=3,
    )
    ax.add_patch(rect)
    # Cell center marker (square = grid cell)
    cx0, cy0 = _xy(lon, lat)
    ax.scatter(
        [cx0], [cy0], s=22, marker="s",
        facecolors=color, edgecolors="white", linewidths=0.5,
        zorder=4,
    )


def _link_color(rec: dict[str, Any]) -> str:
    key = rec.get("dataset_key") or ""
    return DATASET_COLOR.get(key, COL_LINK)


def _draw_link(ax, rec: dict[str, Any]) -> None:
    if not rec["matched"] or rec["facility_lon"] is None:
        return
    cx0, cy0 = _xy(rec["cams_lon"], rec["cams_lat"])
    fx, fy = _xy(rec["facility_lon"], rec["facility_lat"])
    ax.annotate(
        "", xy=(fx, fy), xytext=(cx0, cy0),
        arrowprops=dict(
            arrowstyle="-|>", color=_link_color(rec),
            lw=0.8, alpha=0.75,
            shrinkA=3, shrinkB=4,
        ),
        zorder=5,
    )


def _facility_style(sector_id: str) -> tuple[str, str]:
    override = FACILITY_STYLE.get(sector_id)
    if override:
        return override["marker"], override["color"]
    return _sector_marker(sector_id), _sector_color(sector_id)


def _draw_facility(ax, rec: dict[str, Any]) -> None:
    if not rec["matched"] or rec["facility_lon"] is None:
        return
    fx, fy = _xy(rec["facility_lon"], rec["facility_lat"])
    mk, sector_color = _facility_style(rec["sector_id"])

    ax.scatter(
        [fx], [fy], s=55, c=sector_color,
        edgecolors="white", linewidths=1.0,
        zorder=7, marker=mk,
    )


# ---------------------------------------------------------------------------
# Scale bar & north arrow
# ---------------------------------------------------------------------------

def _add_scale_bar(ax, xmin, xmax, ymin) -> None:
    span_m = xmax - xmin
    # pick a nice round length in km
    target = span_m * 0.18
    nice = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    bar_m = min(nice, key=lambda v: abs(v - target))
    x0 = xmin + (xmax - xmin) * 0.04
    y0 = ymin + (ymax - ymin if False else (ax.get_ylim()[1] - ax.get_ylim()[0])) * 0.04
    ax.plot([x0, x0 + bar_m], [y0, y0], color="black", lw=2.4, zorder=10)
    ax.text(
        x0 + bar_m / 2, y0, f" {bar_m/1000:g} km",
        ha="center", va="bottom", fontsize=7,
        fontweight="bold", zorder=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5),
    )


def _add_north_arrow(ax) -> None:
    ax.annotate(
        "N", xy=(0.97, 0.96), xytext=(0.97, 0.91),
        xycoords="axes fraction",
        ha="center", va="center", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.4),
        zorder=10,
    )


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _write_png(
    points: list[dict[str, Any]],
    bbox: list[float],
    city: str,
    country: str,
    year: int,
    out_path: Path,
    *,
    dpi: int = 180,
) -> Path:
    if not points:
        raise ValueError(
            "No points inside bbox — run proxy point matching for this country first"
        )

    w, s, e, n = bbox
    x0, y0 = _xy(w, s)
    x1, y1 = _xy(e, n)
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)

    # ── stats (CAMS-level for rates, link-level for facility counts) ─────
    cams_points = _cams_summary(points)
    n_total = len(cams_points)
    n_matched = sum(1 for p in cams_points if p["matched"])
    n_unmatched = n_total - n_matched
    match_rate = 100.0 * n_matched / n_total if n_total else 0.0

    by_sector: dict[str, list[dict]] = defaultdict(list)
    for p in cams_points:
        by_sector[p["sector_id"]].append(p)
    by_dataset = Counter(p["dataset_key"] for p in points if p["matched"])

    # ── figure ─────────────────────────────────────────────────────────────
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig = plt.figure(figsize=(13, 9), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3.2, 1],
        height_ratios=[0.10, 1],
        hspace=0.04, wspace=0.04,
        left=0.03, right=0.985, top=0.965, bottom=0.03,
    )
    ax_header = fig.add_subplot(gs[0, :])
    ax = fig.add_subplot(gs[1, 0])
    ax_side = fig.add_subplot(gs[1, 1])

    # ── header banner ──────────────────────────────────────────────────────
    ax_header.axis("off")
    ax_header.set_xlim(0, 1); ax_header.set_ylim(0, 1)
    ax_header.add_patch(mpatches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.02",
        transform=ax_header.transAxes,
        facecolor="#1f2d3d", edgecolor="none",
    ))
    ax_header.text(
        0.012, 0.55,
        f"Point-source matching — {city}, {country}  ·  {year}",
        color="white", fontsize=13, fontweight="bold", va="center",
    )
    # KPI on right
    kpi_x = 0.998
    ax_header.text(
        kpi_x, 0.55,
        f"{n_matched}/{n_total} matched   ·   {match_rate:.0f}% match rate   ·   "
        f"{len(by_sector)} sectors",
        color="#9fd8b8", fontsize=10, fontweight="bold",
        ha="right", va="center",
    )

    # ── map ────────────────────────────────────────────────────────────────
    pad_x = (xmax - xmin) * 0.03
    pad_y = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.8)

    # basemap
    try:
        cx.add_basemap(ax, source=cx.providers.CartoDB.Positron, zoom="auto")
    except Exception:
        try:
            cx.add_basemap(ax, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")
        except Exception:
            ax.set_facecolor("#f2f4f7")

    # bbox rectangle
    ax.plot(
        [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
        color=COL_BBOX, linewidth=1.2, linestyle="--", alpha=0.85, zorder=6,
    )
    ax.text(
        x0, y1, f"  {city} bbox",
        color=COL_BBOX, fontsize=7.5, fontweight="bold",
        va="bottom", ha="left", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
    )

    # draw in z-order: CAMS cells → links → facilities
    for rec in cams_points:
        _draw_cams_cell(ax, rec)
    for rec in points:
        _draw_link(ax, rec)
    for rec in points:
        _draw_facility(ax, rec)

    _add_scale_bar(ax, xmin, xmax, ymin)
    _add_north_arrow(ax)

    # ── side panel: sectors + datasets + symbology ─────────────────────────
    ax_side.axis("off")
    ax_side.set_xlim(0, 1); ax_side.set_ylim(0, 1)

    y_cursor = 0.98
    ax_side.text(
        0.0, y_cursor, "Sectors in view",
        fontsize=10, fontweight="bold", va="top",
    )
    y_cursor -= 0.045

    for sector_id in sorted(by_sector.keys()):
        recs = by_sector[sector_id]
        t = len(recs)
        primary_key = "riurbans"
        n_primary = sum(
            1 for r in recs if r["matched"] and r["dataset_key"] == primary_key
        )
        n_fallback = sum(
            1 for r in recs
            if r["matched"] and r["dataset_key"] != primary_key
        )
        n_matched_sector = n_primary + n_fallback
        n_unmatched_sector = t - n_matched_sector
        rate = 100.0 * n_matched_sector / t if t else 0.0
        color = _sector_color(sector_id)
        mk = _sector_marker(sector_id)
        short = SECTOR_SHORT.get(sector_id, sector_id)

        ax_side.scatter(
            [0.03], [y_cursor - 0.012],
            s=90, c=color, marker=mk, edgecolors="white", linewidths=0.8,
            transform=ax_side.transAxes, zorder=3, clip_on=False,
        )
        ax_side.text(
            0.10, y_cursor - 0.005, short,
            fontsize=8.5, fontweight="bold", va="top",
        )
        ax_side.text(
            0.10, y_cursor - 0.030,
            f"{n_matched_sector}/{t} matched · {rate:.0f}%",
            fontsize=7.2, color="#555", va="top",
        )

        bar_x0, bar_x1 = 0.55, 0.97
        bar_w = bar_x1 - bar_x0
        bar_y = y_cursor - 0.025
        bar_h = 0.012
        ax_side.add_patch(Rectangle(
            (bar_x0, bar_y), bar_w, bar_h,
            facecolor="#eaecef", edgecolor="none",
            transform=ax_side.transAxes,
        ))
        x_off = bar_x0
        if n_primary:
            w_p = bar_w * (n_primary / t)
            ax_side.add_patch(Rectangle(
                (x_off, bar_y), w_p, bar_h,
                facecolor=DATASET_COLOR.get(primary_key, color), edgecolor="none",
                transform=ax_side.transAxes,
            ))
            x_off += w_p
        if n_fallback:
            w_f = bar_w * (n_fallback / t)
            ax_side.add_patch(Rectangle(
                (x_off, bar_y), w_f, bar_h,
                facecolor="#546e7a", edgecolor="none",
                transform=ax_side.transAxes,
            ))
            x_off += w_f
        if n_unmatched_sector:
            w_u = bar_w * (n_unmatched_sector / t)
            ax_side.add_patch(Rectangle(
                (x_off, bar_y), w_u, bar_h,
                facecolor="#d8dce2", edgecolor="none",
                transform=ax_side.transAxes,
            ))

        y_cursor -= 0.075

    ax_side.text(
        0.55, y_cursor + 0.01,
        "Bar: RI-URBANS | fallback | unmatched",
        fontsize=6.2, color="#777", va="bottom",
    )
    y_cursor -= 0.02
    ax_side.text(
        0.0, y_cursor, "Datasets matched",
        fontsize=10, fontweight="bold", va="top",
    )
    y_cursor -= 0.045
    if not by_dataset:
        ax_side.text(0.03, y_cursor, "(none)", fontsize=8, color="#888")
        y_cursor -= 0.04
    else:
        for key, count in by_dataset.most_common():
            label = DATASET_LABEL.get(key, key.upper() or "?")
            color = DATASET_COLOR.get(key, "#444")
            ax_side.add_patch(mpatches.FancyBboxPatch(
                (0.03, y_cursor - 0.022), 0.14, 0.022,
                boxstyle="round,pad=0.005,rounding_size=0.004",
                facecolor=color, edgecolor="none",
                transform=ax_side.transAxes,
            ))
            ax_side.text(
                0.10, y_cursor - 0.011, label,
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
                transform=ax_side.transAxes,
            )
            ax_side.text(
                0.20, y_cursor - 0.011, f"{count} facilities",
                fontsize=8, va="center",
            )
            y_cursor -= 0.040

    # Map symbology (below datasets matched)
    y_cursor -= 0.02
    symbology_items = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COL_MATCHED, markeredgecolor="white",
               markersize=7, label=f"CAMS cell — matched ({n_matched})"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=COL_UNMATCHED, markeredgecolor="white",
               markersize=7, label=f"CAMS cell — unmatched ({n_unmatched})"),
        Line2D([0], [0], color=COL_LINK, linewidth=1.2,
               label="→ matched assignment"),
    ]
    sym_leg = ax_side.legend(
        handles=symbology_items,
        loc="upper left",
        bbox_to_anchor=(0.0, y_cursor),
        bbox_transform=ax_side.transAxes,
        fontsize=7.2, frameon=True, framealpha=0.92, edgecolor="#ccc",
        title="Map symbology", title_fontsize=8,
        handlelength=1.4, labelspacing=0.55,
    )
    sym_leg.get_title().set_fontweight("bold")
    y_cursor -= 0.14

    # ── save ───────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="City point-matching map from Proxy_weights *_matches.json"
    )
    p.add_argument("--city", default="Berlin")
    p.add_argument("--country", default="Germany")
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument(
        "--proxy-root", type=Path,
        default=_REPO / "INPUT" / "Proxy_weights",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="PNG output (default: ./output/{city}_point_match_map.png)",
    )
    args = p.parse_args()

    out = args.output or (
        Path(__file__).resolve().parent / "output"
        / f"{args.city.lower()}_point_match_map.png"
    )

    bbox = _load_bbox(args.city)
    points = _load_sector_points(
        args.proxy_root, DEFAULT_SECTORS, args.country, args.year, bbox,
    )
    path = _write_png(
        points, bbox, args.city, args.country, args.year, out, dpi=args.dpi,
    )
    cams_points = _cams_summary(points)
    n_unmatched = sum(1 for p in cams_points if not p["matched"])
    n_links = sum(1 for p in points if p["matched"])
    print(
        f"Wrote {path} "
        f"({len(cams_points)} CAMS, {n_links} facility links in {args.city} bbox, "
        f"{n_unmatched} unmatched CAMS in view)"
    )


if __name__ == "__main__":
    main()
