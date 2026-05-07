#!/usr/bin/env python3
"""
Interactive Folium map for EMODnet vessel density: GeoTIFF or NetCDF (`vd`).

Supports:
  - GeoTIFF (e.g. EMODnet HA vessel density): band 1 is warped to EPSG:4326 with
    rasterio using the file CRS (no manual x/y interpretation).
  - lon/lat NetCDF: variable `vd` with `latitude`, `longitude`.
  - LAEA NetCDF: `vd` with `x`, `y` in EPSG:3035 (legacy ERDDAP merge).

Usage (from project root):
  python Shipping/Auxiliaries/emodpace_vessel_density_folium_map.py
  python Shipping/Auxiliaries/emodpace_vessel_density_folium_map.py -i data/Shipping/EMODnet/.../file.tif
  python Shipping/Auxiliaries/emodpace_vessel_density_folium_map.py --nc Shipping/emodnet/Cogea_VD_2019_cams_bbox.nc

Requires: xarray, numpy, folium, branca, matplotlib, pyproj, rasterio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_ROOT = Path(__file__).resolve().parents[2]
_AUX = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    _ROOT
    / "data"
    / "Shipping"
    / "EMODnet"
    / "EMODnet_HA_Vessel_Density_allAvg"
    / "vesseldensity_all_2019.tif"
)
DEFAULT_OUT = _AUX / "outputs" / "EMODnet_vessel_density_2019.html"

EMODPACE_NOTICE = (
    "<b>EMODPACE + CAMS crop:</b> this product only overlaps the CAMS grid in a "
    "<b>thin north–south band</b> (~1.3 deg latitude) over the eastern Mediterranean. "
    "The overlay is georeferenced correctly; for <b>full European</b> vessel density "
    "use the Cogea ERDDAP grid (EPSG:3035), e.g. run "
    "<code>data/Shipping/download_cogea_vd_erddap_cams_bbox.py</code> then pass that NetCDF here."
)


def _project_root() -> Path:
    return _ROOT


def _vd_to_float64_seconds(da: xr.DataArray) -> xr.DataArray:
    """Convert vd to float seconds (handles timedelta or float storage)."""
    if np.issubdtype(da.dtype, np.timedelta64):
        return (da / np.timedelta64(1, "s")).astype(np.float64)
    return da.astype(np.float64)


def _apply_fill_mask(vd: xr.DataArray) -> xr.DataArray:
    out = vd
    fv = vd.attrs.get("_FillValue")
    if fv is not None:
        try:
            out = out.where(out != fv)
        except (TypeError, ValueError):
            pass
    return out.where(out != -9999.0, np.nan)


def _aggregate_vd_time_dim(
    vd: xr.DataArray,
    *,
    aggregate: str,
    month_index: int | None,
) -> xr.DataArray:
    if "time" not in vd.dims:
        return vd
    if aggregate == "mean":
        return vd.mean(dim="time", skipna=True)
    if aggregate == "max":
        return vd.max(dim="time", skipna=True)
    if aggregate == "sum":
        return vd.sum(dim="time", skipna=True)
    if aggregate == "month":
        if month_index is None:
            raise ValueError("month_index required for --aggregate month")
        return vd.isel(time=int(month_index))
    raise ValueError(aggregate)


def _prepare_vd_latlon_2d(
    ds: xr.Dataset,
    *,
    aggregate: str,
    month_index: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (data2d, lat_1d, lon_1d) with rows = latitude, cols = longitude.

    Uses xarray transpose/sort so dimension order matches geography (raw numpy
    axis=0 mean assumed time first, which is wrong when ERDDAP order differs).
    """
    vd = _apply_fill_mask(_vd_to_float64_seconds(ds["vd"]))
    vd = _aggregate_vd_time_dim(vd, aggregate=aggregate, month_index=month_index)
    vd = vd.transpose("latitude", "longitude", missing_dims="ignore")
    vd = vd.sortby("latitude").sortby("longitude")
    lat = np.asarray(vd["latitude"].values, dtype=np.float64)
    lon = np.asarray(vd["longitude"].values, dtype=np.float64)
    if lat.ndim != 1 or lon.ndim != 1:
        raise SystemExit("Expected 1D latitude and longitude coordinates.")
    data2d = np.asarray(vd.values, dtype=np.float64)
    return data2d, lat, lon


def _prepare_vd_laea_2d(
    ds: xr.Dataset,
    *,
    aggregate: str,
    month_index: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (vd_2d, y_1d, x_1d) with rows = y, cols = x in EPSG:3035 metres."""
    vd = _apply_fill_mask(_vd_to_float64_seconds(ds["vd"]))
    vd = _aggregate_vd_time_dim(vd, aggregate=aggregate, month_index=month_index)
    vd = vd.transpose("y", "x", missing_dims="ignore")
    vd = vd.sortby("y").sortby("x")
    y = np.asarray(vd["y"].values, dtype=np.float64)
    x = np.asarray(vd["x"].values, dtype=np.float64)
    return np.asarray(vd.values, dtype=np.float64), y, x


def _rgba_from_values(
    data: np.ndarray,
    *,
    legend_scale: str,
    percentile: tuple[float, float],
    auto_robust: bool = True,
) -> tuple[np.ndarray, float, float, str]:
    """
    Return uint8 RGBA (H,W,4), vmin, vmax for the colormap, and a legend note.

    For legend_scale=minmax, vessel-density rasters often have extreme outliers
    (e.g. max >> p98) so naive min–max maps almost everything to the black end of
    inferno. When auto_robust is True, use p2–p98 for scaling if that situation
    is detected; the legend note records full min/max.
    """
    import matplotlib
    import matplotlib.colors as mcolors

    try:
        cmap = matplotlib.colormaps["inferno"]
    except (AttributeError, KeyError):
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("inferno")

    valid = np.isfinite(data)
    legend_note = ""
    if not np.any(valid):
        vmin, vmax = 0.0, 1.0
    elif legend_scale == "minmax":
        vmin_full = float(np.nanmin(data[valid]))
        vmax_full = float(np.nanmax(data[valid]))
        if not np.isfinite(vmin_full) or not np.isfinite(vmax_full):
            vmin, vmax = 0.0, 1.0
        elif vmax_full <= vmin_full:
            vmin, vmax = vmin_full, vmin_full + 1e-6
        else:
            vmin, vmax = vmin_full, vmax_full
            p2 = float(np.nanpercentile(data[valid], 2))
            p98 = float(np.nanpercentile(data[valid], 98))
            if (
                auto_robust
                and p98 > p2
                and vmax_full > max(100.0, p98 * 50.0)
            ):
                vmin, vmax = p2, p98
                legend_note = (
                    f"colors p2–p98 ({vmin:g}–{vmax:g}); full range {vmin_full:g}–{vmax_full:g}"
                )
    else:
        lo, hi = percentile
        vmin = float(np.nanpercentile(data[valid], lo))
        vmax = float(np.nanpercentile(data[valid], hi))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    rgba = cmap(norm(data))
    rgba[~np.isfinite(data), 3] = 0.0
    return (np.clip(rgba * 255, 0, 255).astype(np.uint8), vmin, vmax, legend_note)


def _dedupe_laea_grid(
    y: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collapse duplicate x/y coordinates (e.g. tile merge seams) and average vd.

    RegularGridInterpolator requires strictly monotonic axes; merged ERDDAP tiles
    can repeat the same grid line at boundaries.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if z.shape != (y.size, x.size):
        raise ValueError(f"vd shape {z.shape} does not match y ({y.size}) x ({x.size})")

    y_r = np.round(y, 3)
    x_r = np.round(x, 3)
    y_u, inv_y = np.unique(y_r, return_inverse=True)
    z_y = np.zeros((y_u.size, x.size), dtype=np.float64)
    for i in range(y_u.size):
        z_y[i] = np.nanmean(z[inv_y == i], axis=0)
    x_u, inv_x = np.unique(x_r, return_inverse=True)
    z_out = np.zeros((y_u.size, x_u.size), dtype=np.float64)
    for j in range(x_u.size):
        z_out[:, j] = np.nanmean(z_y[:, inv_x == j], axis=1)
    return y_u, x_u, z_out


def _laea_vd_to_lonlat_grid(
    vd_2d: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    *,
    width_lon: int,
) -> tuple[np.ndarray, float, float, float, float, bool]:
    """
    Resample LAEA (m) `vd` to a regular WGS84 lon/lat grid for Folium.

    Uses rasterio warp (EPSG:3035 -> EPSG:4326) so the output aligns with the
    basemap. Manual lon/lat meshgrid + interpolation can mis-register relative to
    Folium bounds.

    Returns (data2d, lon_min, lon_max, lat_min, lat_max, flip_rgba_for_png).
    flip_rgba_for_png is True when row 0 of data is south (needs flip for Folium
    origin='upper'); False when row 0 is already north (rasterio output).
    """
    from pyproj import Transformer
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import rasterio.transform as rtransform

    y_a, x_a, z = _dedupe_laea_grid(y, x, vd_2d)
    if y_a[0] > y_a[-1]:
        y_a = y_a[::-1]
        z = np.flipud(z)
    if x_a[0] > x_a[-1]:
        x_a = x_a[::-1]
        z = np.fliplr(z)

    height, width = z.shape
    x_min, x_max = float(x_a.min()), float(x_a.max())
    y_min, y_max = float(y_a.min()), float(y_a.max())

    src_transform = from_bounds(x_min, y_min, x_max, y_max, width, height)
    z_src = np.flipud(z).astype(np.float32)

    src_crs = CRS.from_epsg(3035)
    dst_crs = CRS.from_epsg(4326)

    t_f = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lons: list[float] = []
    lats: list[float] = []
    for xi, yi in (
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ):
        lo, la = t_f.transform(xi, yi)
        lons.append(lo)
        lats.append(la)
    lon_min_bb, lon_max_bb = min(lons), max(lons)
    lat_min_bb, lat_max_bb = min(lats), max(lats)
    ny = max(
        2,
        int(
            width_lon
            * (lat_max_bb - lat_min_bb)
            / max(lon_max_bb - lon_min_bb, 1e-9)
        ),
    )

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs,
        dst_crs,
        width,
        height,
        left=x_min,
        bottom=y_min,
        right=x_max,
        top=y_max,
        dst_width=width_lon,
        dst_height=ny,
    )

    dst = np.empty((dst_h, dst_w), dtype=np.float64)
    dst.fill(np.nan)
    reproject(
        source=z_src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    west, south, east, north = rtransform.array_bounds(dst_h, dst_w, dst_transform)
    lon_min, lon_max = west, east
    lat_min, lat_max = south, north

    row0 = dst_transform * (dst_w // 2, 0)
    row1 = dst_transform * (dst_w // 2, dst_h - 1)
    north_at_row0 = row0[1] > row1[1]
    flip_rgba = not north_at_row0

    return dst, lon_min, lon_max, lat_min, lat_max, flip_rgba


def _load_geotiff_wgs84_display(
    tif_path: Path,
    *,
    dst_width: int,
    half_pixel_lat: bool = True,
    lat_offset_deg: float = 0.0,
    lon_offset_deg: float = 0.0,
) -> tuple[np.ndarray, float, float, float, float, bool, str]:
    """
    Warp GeoTIFF band 1 to EPSG:4326 using embedded georeferencing.

    Returns (data2d, lon_min, lon_max, lat_min, lat_max, flip_rgba, legend_note).

    half_pixel_lat: shift bounds south by half a pixel height (degrees). This
    often fixes a slight northward misalignment vs basemaps when cell values
    represent pixel centers and Folium stretches the PNG to outer array_bounds.
    """
    import rasterio
    from rasterio.crs import CRS
    import rasterio.transform as rtransform
    from rasterio.warp import (
        calculate_default_transform,
        reproject,
        Resampling,
        transform_bounds,
    )

    with rasterio.open(tif_path) as src:
        if src.crs is None or not src.crs.is_valid:
            raise SystemExit("GeoTIFF has no valid CRS; cannot align to the map.")
        m = src.read(1, masked=True)
        arr = np.ma.filled(m, np.nan).astype(np.float32)
        arr = np.where(np.abs(arr) > 1e38, np.nan, arr)
        src_transform = src.transform
        height, width = src.height, src.width
        left, bottom, right, top = src.bounds
        src_crs = src.crs
        band_tags = src.tags(1)
        ds_tags = src.tags()

    dst_crs = CRS.from_epsg(4326)
    west, south, east, north = transform_bounds(
        src_crs, dst_crs, left, bottom, right, top
    )
    ny = max(
        2,
        int(dst_width * (north - south) / max(east - west, 1e-9)),
    )

    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs,
        dst_crs,
        width,
        height,
        left=left,
        bottom=bottom,
        right=right,
        top=top,
        dst_width=dst_width,
        dst_height=ny,
    )

    dst = np.empty((dst_h, dst_w), dtype=np.float64)
    dst.fill(np.nan)
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    west, south, east, north = rtransform.array_bounds(dst_h, dst_w, dst_transform)
    lon_min, lon_max = west, east
    lat_min, lat_max = south, north

    row0 = dst_transform * (dst_w // 2, 0)
    row1 = dst_transform * (dst_w // 2, dst_h - 1)
    north_at_row0 = row0[1] > row1[1]
    flip_rgba = not north_at_row0

    if half_pixel_lat and dst_h > 1:
        d_lat = (lat_max - lat_min) / float(dst_h)
        lat_min -= 0.5 * d_lat
        lat_max -= 0.5 * d_lat
    lon_min += lon_offset_deg
    lon_max += lon_offset_deg
    lat_min += lat_offset_deg
    lat_max += lat_offset_deg

    unit = (
        band_tags.get("UNIT")
        or band_tags.get("units")
        or ds_tags.get("UNIT")
        or ""
    )
    legend_note = f" ({unit})" if unit else ""

    return dst, lon_min, lon_max, lat_min, lat_max, flip_rgba, legend_note


def _maybe_emodpace_strip_notice(lat: np.ndarray, lon: np.ndarray) -> str | None:
    """Heuristic: thin EMODPACE ∩ CAMS crop."""
    lat_span = float(np.max(lat) - np.min(lat))
    lon_span = float(np.max(lon) - np.min(lon))
    if lat_span < 4.0 and lon_span > 15.0:
        return EMODPACE_NOTICE
    return None


def _add_fixed_notice(fmap, body_html: str) -> None:
    from branca.element import Element

    div = (
        '<div style="position:fixed;top:10px;left:60px;width:min(440px,92vw);'
        'z-index:10000;background:rgba(255,255,255,0.95);padding:10px 12px;'
        'border:1px solid #444;font-size:12px;line-height:1.35;box-shadow:0 1px 6px rgba(0,0,0,0.25);">'
        f"{body_html}</div>"
    )
    fmap.get_root().html.add_child(Element(div))


def build_map(
    source_path: Path,
    out_html: Path,
    *,
    aggregate: str,
    month_index: int | None,
    legend_scale: str,
    percentile: tuple[float, float],
    opacity: float,
    zoom_start: int | None,
    res_lon: int,
    auto_robust: bool = True,
    geotiff_half_pixel_lat: bool = True,
    geotiff_lat_offset_deg: float = 0.0,
    geotiff_lon_offset_deg: float = 0.0,
    geotiff_mercator: bool = True,
) -> None:
    import folium
    from branca.colormap import LinearColormap

    suffix = source_path.suffix.lower()
    notice: str | None = None
    overlay_name = "Vessel density"
    legend_unit = "vd (seconds)"

    if suffix in (".tif", ".tiff"):
        (
            data2d,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            flip_rgba,
            tif_legend,
        ) = _load_geotiff_wgs84_display(
            source_path,
            dst_width=res_lon,
            half_pixel_lat=geotiff_half_pixel_lat,
            lat_offset_deg=geotiff_lat_offset_deg,
            lon_offset_deg=geotiff_lon_offset_deg,
        )
        bounds = [[lat_min, lon_min], [lat_max, lon_max]]
        overlay_name = "Vessel density (GeoTIFF)"
        legend_unit = f"band 1{tif_legend}"
    else:
        ds = xr.open_dataset(source_path, decode_timedelta=False)
        try:
            if "vd" not in ds.data_vars:
                raise SystemExit("NetCDF has no variable 'vd'.")
            latlon = "latitude" in ds.coords and "longitude" in ds.coords
            if not latlon and not ("x" in ds.coords and "y" in ds.coords):
                raise SystemExit(
                    "Expected either (latitude, longitude) or (x, y) in EPSG:3035 LAEA."
                )
            if latlon:
                data2d, lat, lon = _prepare_vd_latlon_2d(
                    ds,
                    aggregate=aggregate,
                    month_index=month_index,
                )
                notice = _maybe_emodpace_strip_notice(lat, lon)
                lat_min, lat_max = float(lat.min()), float(lat.max())
                lon_min, lon_max = float(lon.min()), float(lon.max())
                bounds = [[lat_min, lon_min], [lat_max, lon_max]]
                flip_rgba = True
            else:
                flat, y, x = _prepare_vd_laea_2d(
                    ds,
                    aggregate=aggregate,
                    month_index=month_index,
                )
                (
                    data2d,
                    lon_min,
                    lon_max,
                    lat_min,
                    lat_max,
                    flip_rgba,
                ) = _laea_vd_to_lonlat_grid(flat, y, x, width_lon=res_lon)
                bounds = [[lat_min, lon_min], [lat_max, lon_max]]
        finally:
            ds.close()

    rgba, vmin, vmax, scale_legend_note = _rgba_from_values(
        data2d,
        legend_scale=legend_scale,
        percentile=percentile,
        auto_robust=auto_robust,
    )
    if flip_rgba:
        rgba = np.flipud(rgba)

    center_lat = 0.5 * (lat_min + lat_max)
    center_lon = 0.5 * (lon_min + lon_max)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start if zoom_start is not None else 8,
        tiles="OpenStreetMap",
    )
    if notice is not None:
        _add_fixed_notice(fmap, notice)
    folium.TileLayer("CartoDB positron", name="Map (light)", control=True).add_to(fmap)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(fmap)

    use_mercator = suffix in (".tif", ".tiff") and geotiff_mercator
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=float(opacity),
        name=overlay_name,
        interactive=True,
        cross_origin=False,
        mercator_project=use_mercator,
    ).add_to(fmap)

    if legend_scale == "minmax":
        leg_caption = f"Vessel density — {legend_unit} — min {vmin:g} to max {vmax:g}"
        if scale_legend_note:
            leg_caption = f"{leg_caption}. {scale_legend_note}"
    else:
        leg_caption = (
            f"Vessel density — {legend_unit} — percentiles "
            f"{percentile[0]:.0f}–{percentile[1]:.0f} of valid cells"
        )
    legend = LinearColormap(
        colors=["#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"],
        vmin=vmin,
        vmax=vmax,
        caption=leg_caption,
    )
    legend.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds(bounds)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_html))


def main() -> int:
    root = _project_root()
    ap = argparse.ArgumentParser(
        description="Folium map for EMODnet vessel density (GeoTIFF or NetCDF vd)."
    )
    try:
        _default_input_hint = str(DEFAULT_INPUT.relative_to(root))
    except ValueError:
        _default_input_hint = str(DEFAULT_INPUT)
    ap.add_argument(
        "-i",
        "--input",
        type=Path,
        default=None,
        help=(
            "GeoTIFF (.tif/.tiff) or NetCDF with variable 'vd'. "
            f"Default: {_default_input_hint}"
        ),
    )
    ap.add_argument(
        "--nc",
        type=Path,
        default=None,
        help="Same as --input (NetCDF); use either -i or --nc.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output HTML path",
    )
    ap.add_argument(
        "--aggregate",
        choices=("mean", "max", "sum", "month"),
        default="mean",
        help="NetCDF only: time aggregation. Ignored for single-band GeoTIFF.",
    )
    ap.add_argument(
        "--month",
        type=int,
        default=None,
        help="0-based month index 0..11 when --aggregate month (default: 0)",
    )
    ap.add_argument(
        "--legend-scale",
        choices=("minmax", "percentile"),
        default="minmax",
        help="Legend and color scale: data min/max (default) or percentiles (--pmin/--pmax)",
    )
    ap.add_argument(
        "--pmin",
        type=float,
        default=2.0,
        help="Lower percentile when --legend-scale percentile (default: 2)",
    )
    ap.add_argument(
        "--pmax",
        type=float,
        default=98.0,
        help="Upper percentile when --legend-scale percentile (default: 98)",
    )
    ap.add_argument("--opacity", type=float, default=0.75, help="Raster overlay opacity")
    ap.add_argument("--zoom", type=int, default=None, help="Initial zoom (default: fit)")
    ap.add_argument(
        "--res-lon",
        type=int,
        default=900,
        help="GeoTIFF/LAEA: output width in pixels after warp to WGS84 (height follows aspect)",
    )
    ap.add_argument(
        "--no-robust",
        action="store_true",
        help="With --legend-scale minmax, use raw min/max only (no p2–p98 fallback for outliers)",
    )
    ap.add_argument(
        "--no-bounds-calibration",
        action="store_true",
        help="GeoTIFF: disable half-pixel latitude shift used to align raster edges with basemap",
    )
    ap.add_argument(
        "--lat-offset",
        type=float,
        default=0.0,
        help="GeoTIFF: add this offset (degrees) to both south and north bounds (positive = north)",
    )
    ap.add_argument(
        "--lon-offset",
        type=float,
        default=0.0,
        help="GeoTIFF: add this offset (degrees) to both west and east bounds (positive = east)",
    )
    ap.add_argument(
        "--no-mercator",
        action="store_true",
        help="GeoTIFF: do not apply Folium mercator_project (plate carrée stretch to bounds instead)",
    )
    args = ap.parse_args()

    src = args.input or args.nc or DEFAULT_INPUT
    src = src if src.is_absolute() else root / src
    out = args.out if args.out.is_absolute() else root / args.out
    if not src.is_file():
        print(f"Input not found: {src}", file=sys.stderr)
        return 1

    month = args.month if args.month is not None else 0
    build_map(
        src,
        out,
        aggregate=args.aggregate,
        month_index=month if args.aggregate == "month" else None,
        legend_scale=args.legend_scale,
        percentile=(args.pmin, args.pmax),
        opacity=args.opacity,
        zoom_start=args.zoom,
        res_lon=args.res_lon,
        auto_robust=not args.no_robust,
        geotiff_half_pixel_lat=not args.no_bounds_calibration,
        geotiff_lat_offset_deg=args.lat_offset,
        geotiff_lon_offset_deg=args.lon_offset,
        geotiff_mercator=not args.no_mercator,
    )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
