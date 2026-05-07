#!/usr/bin/env python3
"""
Quickly visualize reference country boundaries (Eurostat ref-countries) on an interactive map.

By default this looks under:
    <project_root>/data/Ref_Countries
and expects a vector file such as a shapefile, GeoPackage, or GeoJSON.

Usage examples:
    python code/scripts/utilities/visualize_ref_countries.py
    python code/scripts/utilities/visualize_ref_countries.py --path data/Ref_Countries/CNTR_RG_01M_2020_4326.shp
    python code/scripts/utilities/visualize_ref_countries.py --output ref_countries_map.html
"""

import argparse
from pathlib import Path

import geopandas as gpd

try:
    import folium
except ImportError:
    folium = None


def find_default_dataset(ref_dir: Path) -> Path | None:
    """Return the first likely vector dataset in ref_dir, or None."""
    if not ref_dir.is_dir():
        return None

    # Prioritise common vector formats
    extensions = (".shp", ".gpkg", ".geojson", ".json")
    candidates = []
    for ext in extensions:
        candidates.extend(ref_dir.rglob(f"*{ext}"))

    return candidates[0] if candidates else None


def build_default_ref_dir() -> Path:
    """Infer <project_root>/data/Ref_Countries from this script location."""
    here = Path(__file__).resolve()
    # .../code/scripts/utilities -> project_root is two levels up from "code"
    project_root = here.parents[3]
    return project_root / "data" / "Ref_Countries"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize reference country boundaries (Eurostat ref-countries) on a map."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to a vector file (shp/gpkg/geojson). "
             "If omitted, the script searches under <project_root>/data/Ref_Countries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ref_countries_map.html"),
        help="Output HTML map path (default: ref_countries_map.html in current directory).",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default="EPSG:4326",
        help="Target CRS for display (default: EPSG:4326).",
    )
    args = parser.parse_args()

    if folium is None:
        print("Install folium first, e.g.: pip install folium")
        return 1

    ref_dir = build_default_ref_dir()

    if args.path is None:
        dataset_path = find_default_dataset(ref_dir)
        if dataset_path is None:
            print(f"No vector datasets found under {ref_dir}")
            print("Make sure you extracted the Eurostat ref-countries archive "
                  "and point --path to e.g. CNTR_RG_01M_2020_4326.shp.")
            return 1
    else:
        dataset_path = args.path
        if not dataset_path.is_file():
            # Try resolving relative to project ref_dir for convenience
            candidate = ref_dir / dataset_path
            if candidate.is_file():
                dataset_path = candidate
            else:
                print(f"File not found: {dataset_path}")
                return 1

    print(f"Reading {dataset_path}")
    gdf = gpd.read_file(dataset_path)

    # Try to project to WGS84 for web display
    try:
        if gdf.crs is not None and gdf.crs.to_string() != args.crs:
            gdf = gdf.to_crs(args.crs)
    except Exception as exc:
        print(f"Warning: could not reproject to {args.crs}: {exc}")

    if gdf.empty:
        print("Dataset has no geometries.")
        return 1

    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy] in target CRS
    minx, miny, maxx, maxy = bounds
    centre_lat = (miny + maxy) / 2
    centre_lon = (minx + maxx) / 2

    print(f"Extent (CRS {args.crs}):")
    print(f"  minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=4, tiles="OpenStreetMap")
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda _: {"color": "blue", "weight": 1, "fillOpacity": 0.1},
        name="ref_countries",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    folium.LatLngPopup().add_to(m)

    out = args.output.resolve()
    m.save(str(out))
    print(f"Saved map to {out}")

    try:
        import webbrowser

        webbrowser.open(out.as_uri())
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

