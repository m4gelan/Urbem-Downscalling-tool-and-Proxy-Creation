"""CAMS GNFR A area grid: cell IDs on a WGS84 raster, per-cell weight stretch, GeoJSON outlines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def cams_indices(ds: "xr.Dataset") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lon_b = np.asarray(ds["longitude_bounds"].values, dtype=np.float64)
    lat_b = np.asarray(ds["latitude_bounds"].values, dtype=np.float64)
    nlon = int(lon_b.shape[0])
    nlat = int(lat_b.shape[0])
    lon_idx_raw = np.asarray(ds["longitude_index"].values).ravel().astype(np.int64)
    lat_idx_raw = np.asarray(ds["latitude_index"].values).ravel().astype(np.int64)
    if lon_idx_raw.max() >= nlon or lat_idx_raw.max() >= nlat:
        lon_idx_raw = np.maximum(0, lon_idx_raw - 1)
        lat_idx_raw = np.maximum(0, lat_idx_raw - 1)
    lon_ii = np.clip(lon_idx_raw, 0, nlon - 1)
    lat_ii = np.clip(lat_idx_raw, 0, nlat - 1)
    return lon_ii, lat_ii, lon_b, lat_b


def cams_cell_id_grid(
    lons: np.ndarray,
    lats: np.ndarray,
    ds: "xr.Dataset",
    m_area: np.ndarray,
) -> np.ndarray:
    """Map each (lon,lat) sample to CAMS ``source`` index for area cells, or -1."""
    lon_ii, lat_ii, lon_b, lat_b = cams_indices(ds)
    nlon = int(lon_b.shape[0])
    nlat = int(lat_b.shape[0])

    li = np.searchsorted(lon_b[:, 0], lons, side="right") - 1
    li = np.clip(li, 0, nlon - 1)
    ji = np.searchsorted(lat_b[:, 0], lats, side="right") - 1
    ji = np.clip(ji, 0, nlat - 1)
    lw = np.minimum(lon_b[li, 0], lon_b[li, 1])
    le = np.maximum(lon_b[li, 0], lon_b[li, 1])
    ls = np.minimum(lat_b[ji, 0], lat_b[ji, 1])
    ln = np.maximum(lat_b[ji, 0], lat_b[ji, 1])
    valid_lon = (lons >= lw) & (lons <= le)
    valid_lat = (lats >= ls) & (lats <= ln)
    in_bounds = valid_lon & valid_lat

    lookup = np.full(nlon * nlat, -1, dtype=np.int64)
    for i in sorted(int(x) for x in np.flatnonzero(m_area)):
        li_i, ji_i = int(lon_ii[i]), int(lat_ii[i])
        k = li_i * nlat + ji_i
        if lookup[k] < 0:
            lookup[k] = int(i)
    fk = li * nlat + ji
    cid = lookup[fk]
    return np.where(in_bounds & (cid >= 0), cid, -1)


def normalize_weights_per_cams_cell(
    z: np.ndarray,
    cell_id: np.ndarray,
    *,
    base_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-CAMS-cell min–max to [0,1] for display (legacy ``per_cams_cell``)."""
    z = np.asarray(z)
    cell_id = np.asarray(cell_id)
    base_valid = np.asarray(base_valid, dtype=bool)
    if cell_id.shape != z.shape:
        if cell_id.size == z.size:
            cell_id = cell_id.reshape(z.shape)
        else:
            raise ValueError(
                f"cell_id shape {cell_id.shape} does not match z {z.shape}"
            )
    out = np.full(z.shape, np.nan, dtype=np.float64)
    for cid in np.unique(cell_id):
        if cid < 0:
            continue
        m = (cell_id == cid) & base_valid
        if not np.any(m):
            continue
        vals = z[m]
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if hi <= lo:
            out[m] = 0.5
        else:
            out[m] = (z[m] - lo) / (hi - lo)
    disp = base_valid & np.isfinite(out)
    return out, disp


def build_cams_area_grid_geojson(
    ds: "xr.Dataset",
    m_area: np.ndarray,
    bbox_wgs84: tuple[float, float, float, float],
    lon_src: np.ndarray,
    lat_src: np.ndarray,
) -> dict[str, Any]:
    """CAMS area source rectangles (outline / tooltip), clipped to WGS84 bbox."""
    from shapely.geometry import box, mapping

    lon_ii, lat_ii, lon_b, lat_b = cams_indices(ds)
    bw, bs, be, bn = bbox_wgs84
    features: list[dict[str, Any]] = []
    for i in np.flatnonzero(m_area):
        li, ji = int(lon_ii[i]), int(lat_ii[i])
        w, e = float(lon_b[li, 0]), float(lon_b[li, 1])
        s, n = float(lat_b[ji, 0]), float(lat_b[ji, 1])
        if s > n:
            s, n = n, s
        if w > e:
            w, e = e, w
        if e < bw or w > be or n < bs or s > bn:
            continue
        geom = box(w, s, e, n)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "cams_source_index": int(i),
                    "lon_c": float(lon_src[int(i)]),
                    "lat_c": float(lat_src[int(i)]),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}
