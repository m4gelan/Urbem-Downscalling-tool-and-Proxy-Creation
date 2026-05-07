# CAMS area weight proxy — method

Generated (UTC): `2026-04-09T15:21:19.865129+00:00`

## Purpose

Spatial **weights** for downscaling CAMS-REG **area** GNFR A (public power) emissions
onto fine-scale pixels. Emissions are **not** applied here; each CAMS grid cell gets
a set of non-negative weights that sum to 1 over CORINE raster pixels inside that cell.

## Input data

| Item | Role |
|------|------|
| CAMS NetCDF | `C:\Users\leopi\PDM_local\data\given_CAMS\CAMS-REG-ANT_v8.1_TNO_ftp\netcdf\CAMS-REG-v8_1_emissions_year2019.nc` — `longitude_bounds` / `latitude_bounds` define each area cell; `longitude_source` / `latitude_source` are cell centres. |
| CORINE GeoTIFF | `C:\Users\leopi\PDM_local\Input\CORINE\U2018_CLC2018_V2020_20u1.tif` — classification band **1** in the file CRS (e.g. EPSG:3035). |
| LandScan GeoTIFF | `C:\Users\leopi\PDM_local\data\PublicPower\landscan\landscan-global-2020.tif` — population count; warped **bilinear** onto the CORINE grid for each CAMS cell window. |

## Domain filter

WGS84 bounding box (west, south, east, north): `[23.55, 37.85, 23.98, 38.08]`  
Only CAMS area cells whose **rectangle intersects** this box are processed.

## Geometry of each weight row

Each feature is a **polygon**: the CORINE raster pixel footprint (four corners from
the CORINE affine transform, upper-left convention), reprojected to **EPSG:4326** for export.

## Weight construction (per CAMS area cell)

Conceptually:

- **CORINE** defines **where** downscaling is allowed (eligible pixels for public-power
  area mass). Outside the chosen classes, weight is **zero** — CORINE does not scale
  intensity, it **gates** pixels.
- **LandScan** defines **how much** of the cell total each **eligible** pixel receives:
  higher warped population value **P** ⇒ higher raw weight (for default
  `pop_exponent = 1`, weight is proportional to `max(P, pop_floor)`).

Steps in code:

1. Clip CORINE to the CAMS cell polygon in CORINE CRS (`rasterio.mask`).
2. Warp LandScan (bilinear) onto that **same** CORINE grid.
3. Let **P** = LandScan at each pixel, **C** = rounded CORINE class.
4. **Eligibility:** `E = 1` if `C` is in `{121, 3}`, else `E = 0`.
5. **Raw weight:** `w = E * max(P, pop_floor) ** pop_exponent`. So only eligible pixels
   can be positive; among them, larger **P** ⇒ larger **w** (monotone in **P** when
   `pop_exponent > 0`).
6. **Normalize:** `weight_share = w / sum(w)` over the cell (sum of shares = **1**).

**Note:** LandScan is a **count** per its source grid cell, not population **density**
(people per m²). Warping it onto 100 m CORINE pixels with **bilinear** interpolation
tends to **smooth** values: neighbouring pixels get similar **P**, so
`weight_share` can look almost uniform inside a CAMS cell. Use **`nearest`**
resampling (see manifest `landscan_resampling`) for a blockier, more contrasted field.

**`pop_floor` warning:** `w = max(P, pop_floor) ** exponent`. If `pop_floor` is large
(e.g. **1.0**) while many warped **P** are below 1 (common with bilinear fractions),
almost every pixel gets the same raw weight → **nearly uniform shares**. Prefer
`pop_floor=0` unless you deliberately want a minimum weight for low-pop pixels.

### Parameters used

- `corine_codes`: `[121, 3]`
- `corine_band`: 1
- `pop_exponent`: 1.0
- `pop_floor`: 0.0
- `landscan_resampling`: `bilinear`
- `fallback_if_no_corine`: `pop_in_cell`

### Fallbacks (when no pixel matches CORINE codes in the cell)

- **`pop_in_cell`**: Recompute `w` using **all** valid CORINE pixels in the cell (population-only weighting).
- If still no mass: **uniform** over valid CORINE pixels (`uniform_cell`).
- **`skip`**: Omit the cell entirely.

The column `weight_basis` records which case applied.

## Outputs

- GeoJSON: `C:\Users\leopi\PDM_local\PublicPower\outputs\cams_area_weights_athens.geojson` — polygons + attributes.
- Manifest JSON (machine-readable): same stem as GeoJSON with `_manifest.json`.
- This report: same stem with `_report.md`.

## Map

`python PublicPower/Auxiliaries/visualize_cams_area_weights_map.py` loads the manifest
to overlay **LandScan** and **CORINE target-class** rasters for comparison.

