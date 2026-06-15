"""Folium map — 1 km agriculture NMVOC from new UrbEm sector grids."""

from __future__ import annotations

from pathlib import Path

import branca.colormap as bcm
import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from transform import (
    _repo_root,
    domain_box,
    km_sw,
    load_config,
    parse_snap_map,
    read_grid_csv,
    reproject_xy,
    resolve_path,
    source_crs,
)

POL = "NMVOC"
AGRI_SNAP = 10
ZERO_COLOR = "#b8bcc4"


def _cell_starts(lo: float, hi: float, step: int) -> list[int]:
    out: list[int] = []
    x = km_sw(lo, step)
    while x < hi:
        if x >= lo and x + step <= hi:
            out.append(x)
        x += step
    return out


def agriculture_1km_gdf(cfg: dict, root: Path) -> gpd.GeoDataFrame:
    input_dir = resolve_path(cfg["Input_folder"], root)
    sector = parse_snap_map(cfg["SNAP_TO_GNFR"])[AGRI_SNAP]
    sector_dir = input_dir / sector
    if not sector_dir.is_dir():
        raise FileNotFoundError(f"missing sector folder: {sector_dir}")

    step = int(cfg["Grid_step_m"])
    epsg = int(cfg["EPSG"])
    src_crs = source_crs(input_dir)
    df = reproject_xy(read_grid_csv(sector_dir / "area_emission_grid.csv"), src_crs, epsg)
    sub = df.loc[df["pollutant"] == POL].copy()
    if sub.empty:
        raise ValueError(f"no {POL} rows in {sector_dir / 'area_emission_grid.csv'}")

    sub["xcor_sw"] = sub["x"].map(lambda v: km_sw(v, step))
    sub["ycor_sw"] = sub["y"].map(lambda v: km_sw(v, step))
    agg = sub.groupby(["xcor_sw", "ycor_sw"], as_index=False)["emission"].sum()

    if "domain" not in cfg:
        raise KeyError("config missing domain")
    xmin, ymin, xmax, ymax = domain_box(cfg)
    xs = _cell_starts(xmin, xmax, step)
    ys = _cell_starts(ymin, ymax, step)
    if not xs or not ys:
        raise ValueError("domain too small for 1 km grid")

    grid = pd.MultiIndex.from_product([xs, ys], names=["xcor_sw", "ycor_sw"]).to_frame(index=False)
    cells = grid.merge(agg, on=["xcor_sw", "ycor_sw"], how="left")
    cells[POL] = cells["emission"].fillna(0.0)
    cells = cells.drop(columns=["emission"])
    geoms = [
        box(r.xcor_sw, r.ycor_sw, r.xcor_sw + step, r.ycor_sw + step)
        for r in cells.itertuples()
    ]
    gdf = gpd.GeoDataFrame({POL: cells[POL].astype(float).values}, geometry=geoms, crs=f"EPSG:{epsg}")
    return gdf.to_crs(4326)


def render_folium(gdf: gpd.GeoDataFrame, out_path: Path, title: str) -> Path:
    vals = gdf[POL].astype(float)
    pos = vals.loc[vals > 0]
    if pos.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(pos.min())
        vmax = float(pos.max())
        if vmax <= vmin:
            vmax = vmin + 1.0

    colormap = bcm.linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = f"{POL} &gt; 0 (kg/yr per 1 km cell)"

    def cell_style(feat):
        v = float(feat["properties"][POL])
        if v <= 0:
            return {
                "fillColor": ZERO_COLOR,
                "color": "#888888",
                "weight": 0.4,
                "fillOpacity": 0.75,
            }
        return {
            "fillColor": colormap(v),
            "color": "#444444",
            "weight": 0.5,
            "fillOpacity": 0.72,
        }

    west, south, east, north = gdf.total_bounds
    center = [(south + north) / 2.0, (west + east) / 2.0]
    fmap = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap", control_scale=True)

    geojson = gdf.copy()
    geojson["tooltip"] = geojson[POL].map(lambda v: f"{POL}: {v:,.1f} kg/yr")

    folium.GeoJson(
        geojson,
        name=f"Agriculture {POL}",
        style_function=cell_style,
        tooltip=folium.GeoJsonTooltip(fields=["tooltip"], labels=False),
    ).add_to(fmap)

    colormap.add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    title_html = f"""
    <div style="position:fixed;top:12px;left:60px;z-index:9999;background:white;
      padding:8px 12px;border:1px solid #888;font-size:13px;">
      <b>{title}</b><br>1 km aggregated area sources (SNAP 10)<br>
      <span style="color:#666">Grey = zero emission</span>
    </div>"""
    fmap.get_root().html.add_child(folium.Element(title_html))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


def city_label(cfg: dict) -> str:
    name = Path(cfg["Output_folder"]).name
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return str(cfg["City"])


def main():
    root = _repo_root()
    cfg_path = Path(__file__).resolve().parent / "config.yaml"
    cfg = load_config(cfg_path)
    city = city_label(cfg)
    gdf = agriculture_1km_gdf(cfg, root)
    out_dir = resolve_path(cfg["Output_folder"], root) / "figures"
    out_path = out_dir / f"agriculture_{POL}_1km_{city}.html"
    title = f"{city} {cfg['Year']} — agriculture {POL}"
    render_folium(gdf, out_path, title)
    print(
        f"wrote {out_path}  ({len(gdf)} cells, "
        f"{int((gdf[POL] <= 0).sum())} zero, max {gdf[POL].max():,.0f} kg/yr)"
    )


if __name__ == "__main__":
    main()
