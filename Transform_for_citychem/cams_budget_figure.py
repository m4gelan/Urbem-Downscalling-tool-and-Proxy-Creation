"""Two-panel CAMS budget figure: legacy (hashed clip) vs new (green cells)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Transformer
from shapely.geometry import Point, box
import yaml

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from UrbEm_Visualizer.dataset_loaders.cams_emissions import load_cams_area_cells, load_cams_points
from UrbEm_Visualizer.dataset_loaders.cams_grid import _cell_in_domain_envelope, domain_wgs84_from_domain
from UrbEm_Visualizer.dataset_loaders.countries import country_iso3
from UrbEm_Visualizer.downscaling.sector_meta import load_sector_yaml
from UrbEm_Visualizer.paths import project_root

OUT = root / "Output/CityChem/Kozani_2019/figures/cams_budget_illustration.png"
SECTOR = "C_OtherCombustion"
MASS = {"legacy": 167_713, "new_ref": 245_029}
LEGACY_CITY_BOX_32634 = (551000, 4453000, 582000, 4484000)
HATCH = "////"


def _legacy_coarse_cells(w, s, e, n, step_lon=0.1, step_lat=0.05):
    out = []
    lon = w
    while lon < e - 1e-9:
        lon2 = min(lon + step_lon, e)
        lat = s
        while lat < n - 1e-9:
            lat2 = min(lat + step_lat, n)
            out.append(box(lon, lat, lon2, lat2))
            lat = lat2
        lon = lon2
    return out


def _legacy_cell_for_point(lon, lat, w, s, e, n, step_lon=0.1, step_lat=0.05):
    cl = w
    while cl < e - 1e-9:
        cl2 = min(cl + step_lon, e)
        ct = s
        while ct < n - 1e-9:
            ct2 = min(ct + step_lat, n)
            if cl <= lon < cl2 and ct <= lat < ct2:
                return box(cl, ct, cl2, ct2)
            ct = ct2
        cl = cl2
    return None


def _map_extent(*geoms, pad_frac=0.06):
    xs, ys = [], []
    for g in geoms:
        b = g.bounds
        xs.extend([b[0], b[2]])
        ys.extend([b[1], b[3]])
    pad_x = (max(xs) - min(xs)) * pad_frac or 0.01
    pad_y = (max(ys) - min(ys)) * pad_frac or 0.01
    return min(xs) - pad_x, min(ys) - pad_y, max(xs) + pad_x, max(ys) + pad_y


def _draw_map(ax, extent):
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    try:
        cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Positron, zoom="auto")
    except Exception:
        ax.set_facecolor("#eef1f5")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#ccc")


def _plot_clip_parts(ax, geom, city, face, edge, z_in=3, z_out=2):
    """Solid = mass counted inside domain; hatched = same cell, outside domain (excluded)."""
    inside = geom.intersection(city)
    outside = geom.difference(city)
    if not outside.is_empty:
        gpd.GeoDataFrame({"geometry": [outside]}, crs="EPSG:4326").plot(
            ax=ax, facecolor=face, edgecolor=edge, alpha=0.18, linewidth=0.7,
            hatch=HATCH, zorder=z_out,
        )
    if not inside.is_empty:
        gpd.GeoDataFrame({"geometry": [inside]}, crs="EPSG:4326").plot(
            ax=ax, facecolor=face, edgecolor=edge, alpha=0.42, linewidth=0.9, zorder=z_in,
        )


def _draw_edge_inset(fig, example, city, legacy_used, pts_in, pts_out):
    ex_geom, ex_new, ex_full, ex_frac = example
    bx = ex_geom.bounds
    pad_x = (bx[2] - bx[0]) * 1.6 or 0.02
    pad_y = (bx[3] - bx[1]) * 1.6 or 0.02
    ins_ext = (bx[0] - pad_x, bx[1] - pad_y, bx[2] + pad_x, bx[3] + pad_y)
    axins = fig.add_axes([0.68, 0.72, 0.30, 0.24])
    _draw_map(axins, ins_ext)
    _plot_clip_parts(axins, ex_geom, city, "#8fd494", "#2d8a3e", z_in=4, z_out=3)
    gpd.GeoDataFrame({"geometry": [city]}, crs="EPSG:4326").boundary.plot(
        ax=axins, color="#1d3557", linewidth=1.8, zorder=5,
    )
    for lg in legacy_used:
        if lg.intersects(ex_geom):
            _plot_clip_parts(axins, lg, city, "#f4a261", "#c45c26", z_in=3, z_out=2)
    for lon, lat, _ in pts_in:
        if ex_geom.contains(Point(lon, lat)):
            axins.scatter([lon], [lat], s=34, c="#1a7f37", edgecolors="white", linewidths=0.5, zorder=6)
    for lon, lat, _ in pts_out:
        if ex_geom.contains(Point(lon, lat)):
            axins.scatter([lon], [lat], s=26, c="#666", marker="x", linewidths=1.0, zorder=6)
    axins.set_title("Edge example", fontsize=8.5, fontweight="bold", pad=3)
    axins.text(
        0.04, 0.04,
        f"Cell total: {ex_full:,.0f} kg\n"
        f"New: {ex_frac:.0%} in domain ≈ {ex_new:,.0f} kg\n"
        f"Legacy: source point outside box → 0 kg",
        transform=axins.transAxes, fontsize=7, va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.95),
    )


def main():
    run_cfg = yaml.safe_load(open(root / "UrbEm_Visualizer/config/run/Kozani_2019.yaml", encoding="utf-8"))
    domain = run_cfg["domain"]
    w, s, e, n = domain_wgs84_from_domain(domain)
    tr = Transformer.from_crs("EPSG:32634", "EPSG:4326", always_xy=True)
    c = LEGACY_CITY_BOX_32634
    sw = tr.transform(c[0], c[1])
    ne = tr.transform(c[2], c[3])
    city = box(sw[0], sw[1], ne[0], ne[1])

    sec_yaml = load_sector_yaml(SECTOR)
    cams_area = sec_yaml["cams_area_sources"]
    cams_nc = project_root() / run_cfg["paths"]["cams"]
    iso3 = country_iso3(run_cfg["country"])
    year = int(cams_area["year"])
    ec = list(cams_area["emission_category_indices"])
    st = list(cams_area["source_type_indices"])

    cells, _ = load_cams_area_cells(
        cams_nc, year=year, country_iso3=iso3,
        emission_category_indices=ec, source_type_indices=st, pollutants=["NMVOC"],
    )
    points = load_cams_points(
        cams_nc, year=year, country_iso3=iso3,
        emission_category_indices=ec, source_type_indices=st, pollutants=["NMVOC"],
    )

    clip_frac = {}
    clip_path = root / "Output/UrbEm/Kozani_2019/clip_log.json"
    if clip_path.is_file():
        for row in json.loads(clip_path.read_text(encoding="utf-8")):
            if row["sector"] == SECTOR and row["pollutant"] == "NMVOC":
                clip_frac[int(row["cell_id"])] = float(row["clipped_mass_fraction"])

    legacy_used = [g for g in _legacy_coarse_cells(w, s, e, n) if g.intersects(city)]

    official = []
    edge_lost = []
    for cid, row in cells.items():
        if not _cell_in_domain_envelope(row, w, s, e, n):
            continue
        b = row["cell_bounds_wgs84"]
        g = box(b["west"], b["south"], b["east"], b["north"])
        if not g.intersects(city):
            continue
        mass = float(row["pollutants_within_cell"].get("NMVOC", 0.0))
        official.append((g, mass, cid))

    cb = city.bounds
    near = box(cb[0] - 0.04, cb[1] - 0.04, cb[2] + 0.04, cb[3] + 0.04)

    pts_in, pts_out = [], []
    legacy_with_pts = set()
    leg_point_mass = 0.0
    for row in points.values():
        lon, lat = row["longitude"], row["latitude"]
        m = float(row["pollutants"].get("NMVOC", 0.0))
        if m <= 0:
            continue
        p = Point(lon, lat)
        if not near.contains(p):
            continue
        if city.contains(p):
            pts_in.append((lon, lat, m))
            leg_point_mass += m
            lc = _legacy_cell_for_point(lon, lat, w, s, e, n)
            if lc is not None:
                legacy_with_pts.add(lc.wkt)
        else:
            pts_out.append((lon, lat, m))

    for g, mass, cid in official:
        frac = clip_frac.get(cid, 1.0)
        new_contrib = mass * frac
        pt_mass = sum(m for lon, lat, m in pts_in if g.contains(Point(lon, lat)))
        if new_contrib > 80 and pt_mass == 0:
            edge_lost.append((g, new_contrib, mass, frac))

    edge_lost.sort(key=lambda x: -x[1])
    example = edge_lost[0] if edge_lost else None

    extent = _map_extent(city, *[g for g, _, _ in official])
    gdf_city = gpd.GeoDataFrame({"geometry": [city]}, crs="EPSG:4326")
    legacy_active = [g for g in legacy_used if g.wkt in legacy_with_pts]

    fig = plt.figure(figsize=(12.5, 13.5), facecolor="#fafafa")

    # --- Top: legacy ---
    ax1 = fig.add_axes([0.05, 0.52, 0.58, 0.44])
    _draw_map(ax1, extent)

    gdf_off = gpd.GeoDataFrame({"geometry": [g for g, _, _ in official]}, crs="EPSG:4326")
    if not gdf_off.empty:
        gdf_off.plot(ax=ax1, facecolor="none", edgecolor="#2e4db0", linewidth=1.0, alpha=0.85, zorder=1)

    for lg in legacy_used:
        if lg.wkt not in legacy_with_pts:
            gpd.GeoDataFrame({"geometry": [lg]}, crs="EPSG:4326").plot(
                ax=ax1, facecolor="none", edgecolor="#c45c26", linewidth=0.9, alpha=0.5, zorder=2,
            )
    for lg in legacy_active:
        _plot_clip_parts(ax1, lg, city, "#f4a261", "#c45c26", z_in=4, z_out=3)

    if pts_in:
        ax1.scatter([p[0] for p in pts_in], [p[1] for p in pts_in], s=24, c="#1a7f37",
                    edgecolors="white", linewidths=0.5, zorder=7)
    if pts_out:
        ax1.scatter([p[0] for p in pts_out], [p[1] for p in pts_out], s=18, c="#666",
                    marker="x", linewidths=1.0, zorder=6)
    gdf_city.boundary.plot(ax=ax1, color="#1d3557", linewidth=2.2, zorder=8)

    ax1.set_title(
        f"Legacy — synthetic cells + CAMS source points  |  cams_stat ≈ {MASS['legacy']:,} kg NMVOC",
        fontsize=10, fontweight="bold", pad=6,
    )

    if example is not None:
        _draw_edge_inset(fig, example, city, legacy_used, pts_in, pts_out)

    leg_top = [
        mpatches.Patch(facecolor="none", edgecolor="#1d3557", linewidth=2.2, label="City domain"),
        mpatches.Patch(facecolor="none", edgecolor="#2e4db0", linewidth=1.0, label="Official CAMS-REG cells"),
        mpatches.Patch(facecolor="none", edgecolor="#c45c26", linewidth=1.0, label="Legacy synthetic cells"),
        mpatches.Patch(facecolor="#f4a261", edgecolor="#c45c26", alpha=0.42, label="Legacy cell portion counted (inside domain)"),
        mpatches.Patch(facecolor="#f4a261", edgecolor="#c45c26", alpha=0.18, hatch=HATCH, label="Legacy cell portion excluded (outside domain)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a7f37", markersize=6,
                   label="Source point inside city box"),
        plt.Line2D([0], [0], marker="x", color="#666", markersize=6, linestyle="None",
                   label="Source point outside city box"),
    ]
    fig.legend(handles=leg_top, loc="upper left", bbox_to_anchor=(0.68, 0.68),
               fontsize=7.5, frameon=True, facecolor="white", edgecolor="#ddd", title="Legacy panel")

    # --- Bottom: new pipeline ---
    ax2 = fig.add_axes([0.05, 0.06, 0.58, 0.44])
    _draw_map(ax2, extent)

    for g, _, _ in official:
        _plot_clip_parts(ax2, g, city, "#8fd494", "#2d8a3e", z_in=3, z_out=2)

    gdf_city.boundary.plot(ax=ax2, color="#1d3557", linewidth=2.2, zorder=5)

    ax2.set_title(
        f"New pipeline — official CAMS-REG cells  |  reference ≈ {MASS['new_ref']:,} kg NMVOC (prorated)",
        fontsize=10, fontweight="bold", pad=6,
    )

    leg_bot = [
        mpatches.Patch(facecolor="none", edgecolor="#1d3557", linewidth=2.2, label="City domain"),
        mpatches.Patch(facecolor="#8fd494", edgecolor="#2d8a3e", alpha=0.42, label="CAMS cell mass counted (inside domain)"),
        mpatches.Patch(facecolor="#8fd494", edgecolor="#2d8a3e", alpha=0.18, hatch=HATCH, label="CAMS cell portion excluded (outside domain)"),
    ]
    fig.legend(handles=leg_bot, loc="upper left", bbox_to_anchor=(0.68, 0.38),
               fontsize=7.5, frameon=True, facecolor="white", edgecolor="#ddd", title="New panel")

    fig.text(
        0.05, 0.015,
        f"Solid fill = mass included in budget for that cell; hatched = same cell extends outside the city box and that share is excluded. "
        f"Legacy only enters cells with a source point inside the box ({len(pts_in)} points). "
        f"New uses all {len(official)} overlapping official cells.",
        fontsize=8, color="#333",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="#fafafa")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
