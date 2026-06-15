"""Compare legacy vs new emission extents in EPSG:3035."""

from __future__ import annotations

from pathlib import Path

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from pyproj import Transformer
from shapely.geometry import box

from transform import _repo_root, load_config, resolve_path, source_crs

TARGET_CRS = "EPSG:3035"


def _bbox_3035(xs, ys, src_epsg: int) -> tuple[float, float, float, float]:
    minx, maxx = float(min(xs)), float(max(xs))
    miny, maxy = float(min(ys)), float(max(ys))
    tr = Transformer.from_crs(f"EPSG:{src_epsg}", TARGET_CRS, always_xy=True)
    tx, ty = zip(*[tr.transform(x, y) for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))])
    return min(tx), min(ty), max(tx), max(ty)


def _add_coords(xs, ys, df: pd.DataFrame, pairs: list[tuple[str, str]], *, skip_zero: bool = False):
    for _, row in df.iterrows():
        for xk, yk in pairs:
            x, y = float(row[xk]), float(row[yk])
            if skip_zero and x == 0.0 and y == 0.0:
                continue
            xs.append(x)
            ys.append(y)


def legacy_kozani_bbox(legacy_dir: Path, src_epsg: int) -> tuple[float, float, float, float]:
    xs, ys = [], []
    areas_path = next(legacy_dir.glob("*_areas_sources_*.csv"))
    stem = areas_path.name.replace(".csv", "")
    tag, year_s = stem.rsplit("_areas_sources_", 1)
    year = int(year_s)
    _add_coords(xs, ys, pd.read_csv(legacy_dir / f"{tag}_areas_sources_{year}.csv"), [("xcor_sw", "ycor_sw"), ("xcor_ne", "ycor_ne")])
    pts = legacy_dir / f"{tag}_point_sources_{year}.csv"
    if pts.is_file():
        _add_coords(xs, ys, pd.read_csv(pts), [("xcor", "ycor")])
    lines = legacy_dir / f"{tag}_lines_sources_{year}_all_increase.csv"
    if lines.is_file():
        _add_coords(
            xs, ys, pd.read_csv(lines),
            [("xcor_start", "ycor_start"), ("xcor_end", "ycor_end")],
            skip_zero=True,
        )
    if not xs:
        raise ValueError(f"no legacy coordinates in {legacy_dir}")
    return _bbox_3035(xs, ys, src_epsg)


def legacy_athens_bbox(legacy_dir: Path, src_epsg: int) -> tuple[float, float, float, float]:
    xs, ys = [], []
    _add_coords(xs, ys, pd.read_csv(legacy_dir / "Ath19_GNFRareas.csv"), [("xcor_sw", "ycor_sw"), ("xcor_ne", "ycor_ne")])
    _add_coords(xs, ys, pd.read_csv(legacy_dir / "Ath19_GNFRpoints.csv"), [("xcor", "ycor")])
    _add_coords(
        xs, ys, pd.read_csv(legacy_dir / "Ath19_GNFRlinesx1_tertiary.csv"),
        [("xcor_start", "ycor_start"), ("xcor_end", "ycor_end")],
    )
    return _bbox_3035(xs, ys, src_epsg)


def new_citychem_bbox(output_dir: Path, city: str, src_epsg: int) -> tuple[float, float, float, float]:
    xs, ys = [], []
    _add_coords(xs, ys, pd.read_csv(output_dir / f"area_source_{city}.csv"), [("xcor_sw", "ycor_sw"), ("xcor_ne", "ycor_ne")])
    _add_coords(xs, ys, pd.read_csv(output_dir / f"line_source_{city}.csv"), [("xcor_start", "ycor_start"), ("xcor_end", "ycor_end")])
    _add_coords(xs, ys, pd.read_csv(output_dir / f"point_source_{city}.csv"), [("xcor", "ycor")])
    return _bbox_3035(xs, ys, src_epsg)


def urbem_domain_bbox(input_dir: Path) -> tuple[float, float, float, float] | None:
    manifest = input_dir / "manifest.yaml"
    if not manifest.is_file():
        return None
    with open(manifest, encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    d = meta.get("domain") or (meta.get("config") or {}).get("domain")
    if not d:
        return None
    return float(d["xmin"]), float(d["ymin"]), float(d["xmax"]), float(d["ymax"])


def legacy_bbox(cfg: dict, root: Path, src_epsg: int) -> tuple[float, float, float, float]:
    city = str(cfg["City"])
    year = int(cfg["Year"])
    nested = root / "Output" / "OLD" / city.upper() / "Emissions" / str(year) / "V3" / "Increase_Factor_1" / "Results" / "Results_CSVs"
    if nested.is_dir():
        return legacy_kozani_bbox(nested, src_epsg)
    flat = root / "Output" / "OLD" / city.upper()
    if (flat / "Ath19_GNFRareas.csv").is_file():
        return legacy_athens_bbox(flat, src_epsg)
    raise FileNotFoundError(f"no legacy CSVs for {city} under Output/OLD")


def _gdf_from_bbox(b: tuple[float, float, float, float], label: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"label": [label]}, geometry=[box(*b)], crs=TARGET_CRS)


def plot_bbox_map(
    legacy_b: tuple[float, float, float, float],
    new_b: tuple[float, float, float, float],
    domain_b: tuple[float, float, float, float] | None,
    city: str,
    out_path: Path,
):
    frames = [
        _gdf_from_bbox(legacy_b, "legacy"),
        _gdf_from_bbox(new_b, "new CityChem"),
    ]
    if domain_b is not None:
        frames.append(_gdf_from_bbox(domain_b, "UrbEm domain"))

    all_b = [legacy_b, new_b] + ([domain_b] if domain_b else [])
    minx = min(b[0] for b in all_b)
    miny = min(b[1] for b in all_b)
    maxx = max(b[2] for b in all_b)
    maxy = max(b[3] for b in all_b)
    pad_x = (maxx - minx) * 0.12 or 5000.0
    pad_y = (maxy - miny) * 0.12 or 5000.0

    fig, ax = plt.subplots(figsize=(10, 10))
    styles = {
        "legacy": {"edgecolor": "#e05c5c", "facecolor": "none", "linewidth": 2.5, "linestyle": "-"},
        "new CityChem": {"edgecolor": "#4f7cff", "facecolor": "none", "linewidth": 2.5, "linestyle": "-"},
        "UrbEm domain": {"edgecolor": "#3ecf8e", "facecolor": "none", "linewidth": 2.0, "linestyle": "--"},
    }
    for gdf in frames:
        label = gdf["label"].iloc[0]
        gdf.boundary.plot(ax=ax, **styles[label], label=label, zorder=3)

    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect("equal")
    try:
        cx.add_basemap(ax, crs=TARGET_CRS, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")
    except Exception:
        ax.set_facecolor("#eef1f5")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title(f"{city} bounding boxes ({TARGET_CRS})", fontsize=12)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    def fmt(b, name):
        return (
            f"{name}: xmin={b[0]:.1f} ymin={b[1]:.1f} "
            f"xmax={b[2]:.1f} ymax={b[3]:.1f} "
            f"({(b[2]-b[0])/1000:.1f} x {(b[3]-b[1])/1000:.1f} km)"
        )

    notes = [fmt(legacy_b, "Legacy"), fmt(new_b, "New")]
    if domain_b:
        notes.append(fmt(domain_b, "UrbEm domain"))
    ax.text(
        0.02, 0.02, "\n".join(notes), transform=ax.transAxes, fontsize=8,
        va="bottom", ha="left", bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    for line in notes:
        print(line)


def main():
    root = _repo_root()
    cfg = load_config(Path(__file__).resolve().parent / "config.yaml")
    city = str(cfg["City"])
    src_epsg = int(cfg["EPSG"])
    input_dir = resolve_path(cfg["Input_folder"], root)
    output_dir = resolve_path(cfg["Output_folder"], root)
    out_path = output_dir / "figures" / f"bbox_compare_{city}_3035.png"

    legacy_b = legacy_bbox(cfg, root, src_epsg)
    new_b = new_citychem_bbox(output_dir, city, src_epsg)
    domain_b = urbem_domain_bbox(input_dir)
    plot_bbox_map(legacy_b, new_b, domain_b, city, out_path)


if __name__ == "__main__":
    main()
