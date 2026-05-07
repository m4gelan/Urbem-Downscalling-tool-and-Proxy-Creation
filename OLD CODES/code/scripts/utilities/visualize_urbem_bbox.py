#!/usr/bin/env python3
"""
Visualize UrbEm domain bounding box (UTM) on an interactive map.

Reads xmin, xmax, ymin, ymax in UTM (default EPSG:32634) and opens a Folium
HTML map with the rectangle. Optionally pass coordinates as arguments.

Usage:
    python code/scripts/utilities/visualize_urbem_bbox.py
    python code/scripts/utilities/visualize_urbem_bbox.py --xmin 716397 --xmax 761397 --ymin 4191261 --ymax 4236261 --epsg 32634
"""

import argparse
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

try:
    import folium
except ImportError:
    folium = None


def utm_bbox_to_wgs84(xmin: float, xmax: float, ymin: float, ymax: float, epsg: int = 32634):
    """Convert UTM bbox to WGS84 polygon and centre."""
    geom = box(xmin, ymin, xmax, ymax)
    gdf = gpd.GeoDataFrame([1], geometry=[geom], crs=f"EPSG:{epsg}")
    gdf_wgs = gdf.to_crs("EPSG:4326")
    bounds = gdf_wgs.total_bounds
    lon_min, lat_min, lon_max, lat_max = bounds
    centre_lat = (lat_min + lat_max) / 2
    centre_lon = (lon_min + lon_max) / 2
    return gdf_wgs, centre_lat, centre_lon, (lat_min, lat_max, lon_min, lon_max)


def main():
    parser = argparse.ArgumentParser(description="Visualize UrbEm UTM bbox on a map.")
    parser.add_argument("--xmin", type=float, default=716397)
    parser.add_argument("--xmax", type=float, default=761397)
    parser.add_argument("--ymin", type=float, default=4191261)
    parser.add_argument("--ymax", type=float, default=4236261)
    parser.add_argument("--epsg", type=int, default=32634, help="UTM zone (e.g. 32634)")
    parser.add_argument("--output", type=Path, default=None, help="HTML path (default: bbox_map.html in cwd)")
    parser.add_argument("--no-open", action="store_true", help="Do not open the HTML in browser")
    args = parser.parse_args()

    if folium is None:
        print("Install folium: pip install folium")
        return 1

    gdf_wgs, lat_c, lon_c, (lat_min, lat_max, lon_min, lon_max) = utm_bbox_to_wgs84(
        args.xmin, args.xmax, args.ymin, args.ymax, args.epsg
    )
    bounds_wgs = [[lat_min, lon_min], [lat_max, lon_max]]

    m = folium.Map(location=[lat_c, lon_c], zoom_start=10, tiles="OpenStreetMap")
    folium.GeoJson(
        gdf_wgs.__geo_interface__,
        style_function=lambda _: {"color": "red", "weight": 3, "fillOpacity": 0.1},
    ).add_to(m)
    folium.Rectangle(bounds=bounds_wgs, color="red", weight=2, fill_opacity=0.05).add_to(m)
    folium.LatLngPopup().add_to(m)

    out = args.output or Path("bbox_map.html")
    out = Path(out)
    m.save(str(out))
    print(f"Saved {out.absolute()}")
    print(f"Centre (WGS84): {lat_c:.5f}, {lon_c:.5f}")

    if not args.no_open:
        import webbrowser
        webbrowser.open(out.as_uri())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
