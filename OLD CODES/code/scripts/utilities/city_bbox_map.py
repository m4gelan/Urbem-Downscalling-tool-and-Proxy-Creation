#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a bounding box around a city and visualize it on an interactive map.

Default output is an interactive HTML map (Folium / Leaflet.js) with crisp
OpenStreetMap tiles at every zoom level, the bounding box, and a CAMS-REG
grid overlay (0.05 lat x 0.1 lon).  Use --static for a matplotlib PNG instead.

Output files are named  Border_box_map_of_{City}.html  and  .png  by default.
The PNG is always generated alongside the HTML (with OSM tiles when online).

Usage:
    python city_bbox_map.py "Kozani"                       # HTML + PNG, opens in browser
    python city_bbox_map.py "Kozani" --no-show             # save only, don't open
    python city_bbox_map.py "Ioannina" --radius-km 30
    python city_bbox_map.py "Kozani" --static              # PNG only (no HTML)
"""

from __future__ import annotations

import argparse
import math
import sys
import webbrowser
from pathlib import Path

try:
    from OSMPythonTools.nominatim import Nominatim
except ImportError:
    Nominatim = None

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    folium = None

METRES_PER_DEG_LAT = 111_320
METRES_PER_DEG_LON_AT_EQUATOR = 111_320

# CAMS-REG grid resolution (degrees)
CAMS_DLAT = 0.05
CAMS_DLON = 0.1


# ---------------------------------------------------------------------------
# Geocoding & bbox
# ---------------------------------------------------------------------------

def metres_to_deg_lat(m: float) -> float:
    return m / METRES_PER_DEG_LAT


def metres_to_deg_lon(m: float, lat_deg: float) -> float:
    return m / (METRES_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat_deg)))


def geocode(city_name: str) -> tuple[float, float, str] | None:
    """Return (lat, lon, display_name) or None."""
    if Nominatim is None:
        print("Error: OSMPythonTools is required.  pip install OSMPythonTools")
        return None
    result = Nominatim().query(city_name)
    if result is None or not result.toJSON():
        return None
    data = result.toJSON()
    if not data:
        return None
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    display = data[0].get("display_name", city_name)
    return (lat, lon, display)


def bbox_from_centre(
    lat: float,
    lon: float,
    radius_km: float | None = None,
    margin_deg: float | None = None,
) -> tuple[float, float, float, float]:
    """Return (lon_min, lon_max, lat_min, lat_max) in WGS84."""
    if radius_km is not None:
        half_lat = metres_to_deg_lat(radius_km * 1000)
        half_lon = metres_to_deg_lon(radius_km * 1000, lat)
    elif margin_deg is not None:
        half_lat = margin_deg
        half_lon = margin_deg
    else:
        half_lat = half_lon = 0.1
    return (lon - half_lon, lon + half_lon, lat - half_lat, lat + half_lat)


# ---------------------------------------------------------------------------
# Interactive HTML map (Folium / Leaflet)
# ---------------------------------------------------------------------------

def _cams_grid_geojson(lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> dict:
    """Build a GeoJSON FeatureCollection of CAMS-REG grid lines covering the bbox."""
    features = []
    # Vertical lines (constant lon, every CAMS_DLON)
    lon = math.floor(lon_min / CAMS_DLON) * CAMS_DLON
    while lon <= lon_max + 1e-9:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon, lat_min], [lon, lat_max]],
            },
            "properties": {"type": "cams_grid"},
        })
        lon += CAMS_DLON
    # Horizontal lines (constant lat, every CAMS_DLAT)
    lat = math.floor(lat_min / CAMS_DLAT) * CAMS_DLAT
    while lat <= lat_max + 1e-9:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[lon_min, lat], [lon_max, lat]],
            },
            "properties": {"type": "cams_grid"},
        })
        lat += CAMS_DLAT
    return {"type": "FeatureCollection", "features": features}


def build_interactive_map(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    city_name: str,
    centre_lat: float,
    centre_lon: float,
) -> "folium.Map":
    """Return a Folium Map with bbox rectangle, CAMS grid, and multiple tile layers."""
    centre = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]

    m = folium.Map(
        location=centre,
        zoom_start=10,
        tiles=None,
        control_scale=True,
    )

    # Multiple tile layers the user can switch between
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite",
        max_zoom=18,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="OpenTopoMap",
        max_zoom=17,
    ).add_to(m)

    # Bounding box
    bbox_coords = [
        [lat_min, lon_min],
        [lat_min, lon_max],
        [lat_max, lon_max],
        [lat_max, lon_min],
        [lat_min, lon_min],
    ]
    folium.PolyLine(
        locations=bbox_coords,
        color="red",
        weight=3,
        opacity=0.9,
        tooltip=f"Bounding box ({lon_max - lon_min:.3f} x {lat_max - lat_min:.3f} deg)",
    ).add_to(m)

    # CAMS grid as a GeoJSON overlay (toggleable)
    cams_geojson = _cams_grid_geojson(lon_min, lon_max, lat_min, lat_max)
    folium.GeoJson(
        cams_geojson,
        name=f"CAMS grid ({CAMS_DLAT} x {CAMS_DLON} deg)",
        style_function=lambda _: {
            "color": "#00bcd4",
            "weight": 1.0,
            "opacity": 0.55,
            "dashArray": "4 3",
        },
    ).add_to(m)

    # City centre marker
    folium.Marker(
        location=[centre_lat, centre_lon],
        popup=f"<b>{city_name}</b><br>lat {centre_lat:.5f}, lon {centre_lon:.5f}",
        tooltip=city_name,
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # Corner labels showing bbox coordinates
    for lat, lon, label in [
        (lat_min, lon_min, f"{lat_min:.4f}, {lon_min:.4f}"),
        (lat_min, lon_max, f"{lat_min:.4f}, {lon_max:.4f}"),
        (lat_max, lon_min, f"{lat_max:.4f}, {lon_min:.4f}"),
        (lat_max, lon_max, f"{lat_max:.4f}, {lon_max:.4f}"),
    ]:
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="red",
            fill=True,
            fill_opacity=0.8,
            tooltip=label,
        ).add_to(m)

    # Fit the map to the bbox with some padding
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]], padding=[30, 30])

    # Layer control toggle
    folium.LayerControl(collapsed=False).add_to(m)

    return m


# ---------------------------------------------------------------------------
# Static matplotlib map (--static fallback)
# ---------------------------------------------------------------------------

FIG_SIZE = (14, 11)
SAVE_DPI = 300

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False
    ctx = None


def _has_tile_network(host: str = "tile.openstreetmap.org", port: int = 443, timeout: float = 3.0) -> bool:
    import socket
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


def plot_static_map(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    city_name: str,
    centre_lat: float,
    centre_lon: float,
    save_path: Path | None = None,
    show: bool = True,
    dpi: int = SAVE_DPI,
) -> None:
    """High-quality static PNG with OSM basemap (contextily), CAMS grid, and bbox."""
    try:
        if not show:
            import matplotlib
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        print("Error: matplotlib is required for --static.  pip install matplotlib")
        return

    pad = 0.15 * max(lon_max - lon_min, lat_max - lat_min)
    extent = [lon_min - pad, lon_max + pad, lat_min - pad, lat_max + pad]

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, facecolor="white")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # Try to add OSM basemap tiles via contextily
    tiles_ok = False
    if HAS_CONTEXTILY and _has_tile_network():
        try:
            ctx.add_basemap(
                ax,
                crs="EPSG:4326",
                source=ctx.providers.OpenStreetMap.Mapnik,
                zoom=11,
                attribution=False,
            )
            tiles_ok = True
        except Exception as e:
            print(f"Tile download failed ({e}), using plain background.")
    if not tiles_ok:
        ax.set_facecolor("white")

    # CAMS grid
    lon = math.floor(extent[0] / CAMS_DLON) * CAMS_DLON
    while lon <= extent[1] + 1e-9:
        ax.axvline(lon, color="#00bcd4", linewidth=0.7, alpha=0.6, zorder=3)
        lon += CAMS_DLON
    lat = math.floor(extent[2] / CAMS_DLAT) * CAMS_DLAT
    while lat <= extent[3] + 1e-9:
        ax.axhline(lat, color="#00bcd4", linewidth=0.7, alpha=0.6, zorder=3)
        lat += CAMS_DLAT

    # Bounding box
    rect = mpatches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=3,
        edgecolor="red",
        facecolor="none",
        zorder=5,
    )
    ax.add_patch(rect)

    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"Bounding box: {city_name}", fontsize=14, fontweight="bold")
    ax.legend(
        handles=[
            Line2D([0], [0], color="red", linewidth=2.5, label="Bounding box"),
            Line2D([0], [0], color="#00bcd4", linewidth=0.8, alpha=0.6,
                   label=f"CAMS grid ({CAMS_DLAT} x {CAMS_DLON} deg)"),
        ],
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
    )
    if not tiles_ok:
        ax.grid(True, alpha=0.3, linestyle="--", zorder=1)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if save_path:
        fig.canvas.draw()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print("Saved:", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Geocode a city, create a bounding box, and show it on an interactive map.",
    )
    parser.add_argument(
        "city",
        type=str,
        help="City name (e.g. 'Ioannina', 'Kozani, Greece')",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--radius-km",
        type=float,
        default=None,
        help="Half-side of the box in km (e.g. 25 for a ~50 km box)",
    )
    group.add_argument(
        "--margin-deg",
        type=float,
        default=None,
        help="Half-side of the box in degrees (e.g. 0.15)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save to this path (.html for interactive, .png for static)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the result in the browser / plot window",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Generate a static matplotlib PNG instead of an interactive HTML map",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=SAVE_DPI,
        metavar="N",
        help="DPI for static PNG (default %s)" % SAVE_DPI,
    )
    args = parser.parse_args()

    if not args.radius_km and not args.margin_deg:
        args.radius_km = 25.0
        print("Using default --radius-km 25 (override with --radius-km or --margin-deg)")

    result = geocode(args.city)
    if result is None:
        print("Could not find location for:", args.city)
        return 1
    lat, lon, display_name = result
    print("Location:", display_name)
    print(f"Centre (WGS84): lat = {lat}, lon = {lon}")

    lon_min, lon_max, lat_min, lat_max = bbox_from_centre(
        lat, lon,
        radius_km=args.radius_km,
        margin_deg=args.margin_deg,
    )
    print(f"Bounding box: lon_min={lon_min:.5f}  lon_max={lon_max:.5f}  "
          f"lat_min={lat_min:.5f}  lat_max={lat_max:.5f}")

    # Build canonical filename stem
    city_clean = args.city.split(",")[0].strip().replace(" ", "_")
    default_stem = f"Border_box_map_of_{city_clean}"

    if args.static:
        save_path = args.save
        if save_path is None:
            save_path = Path("results") / city_clean / f"{default_stem}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot_static_map(
            lon_min, lon_max, lat_min, lat_max,
            args.city, lat, lon,
            save_path=save_path,
            show=not args.no_show,
            dpi=args.dpi,
        )
        return 0

    # Interactive HTML map (default)
    if not HAS_FOLIUM:
        print("Error: folium is required for interactive maps.  pip install folium")
        print("       Use --static for a matplotlib PNG fallback.")
        return 1

    m = build_interactive_map(
        lon_min, lon_max, lat_min, lat_max,
        args.city, lat, lon,
    )

    # Determine save path
    save_path = args.save
    if save_path is None:
        save_path = Path("results") / city_clean / f"{default_stem}.html"
    if save_path.suffix.lower() not in (".html", ".htm"):
        save_path = save_path.with_suffix(".html")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(save_path))
    print("Saved:", save_path)

    # Also generate a high-resolution PNG alongside the HTML
    png_path = save_path.with_suffix(".png")
    print(f"Generating static PNG ({SAVE_DPI} DPI): {png_path}")
    plot_static_map(
        lon_min, lon_max, lat_min, lat_max,
        args.city, lat, lon,
        save_path=png_path,
        show=False,
        dpi=args.dpi,
    )

    if not args.no_show:
        webbrowser.open(save_path.resolve().as_uri())
        print("Opened in browser.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
