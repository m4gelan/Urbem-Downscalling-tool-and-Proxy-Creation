# CAMS J_Waste within-cell weight rasters

Python workflow to build **fine-grid spatial weights** for CAMS-REG-ANT **GNFR J (Waste) area** emissions: for each native CAMS grid cell, weights over fine pixels **sum to 1**. This package does **not** compute downscaled emissions.

## Run

From the repository root (`PDM_local`), with dependencies installed (`rasterio`, `geopandas`, `xarray`, `pandas`, `numpy`, `shapely`, `pyyaml`, `openpyxl`; optional `scipy` for Gaussian smoothing of WWTP footprints):

```bash
python -m Waste.j_waste_weights.main
python -m Waste.j_waste_weights.main --config Waste/j_waste_weights/config.yaml
```

Edit paths in [`config.yaml`](config.yaml) under `paths:` and `output:`.

## Outputs (default `SourceProxies/outputs/EL/`)

| File | Description |
|------|-------------|
| `cams_j_waste_within_cell_weights.tif` | Multi-band float32; band = pollutant; within-CAMS-cell weights |
| `pollutant_band_mapping.csv` | Band index ↔ pollutant name |
| `country_pollutant_subsector_weights.csv` | CEIP-derived `E_*`, `w_*`, fallback metadata |
| `diagnostics_zero_proxy_cells.csv` | CAMS cells where uniform fallback was used |
| `diagnostics_ceip_fallbacks.csv` | CEIP weight fallback tier log |

Optional intermediates when `output.write_intermediates: true`: `proxy_solid.tif`, `proxy_wastewater.tif`, `proxy_residual.tif`, `composite_proxy_<pollutant>.tif`, `diagnostic_imperviousness_valid.tif`.

## GIS semantics (short)

- **CAMS cell id:** Fine pixel **center** (EPSG:3035 → WGS84) must lie inside `longitude_bounds[i]` and `latitude_bounds[j]` from the CAMS NetCDF. Id = `ji * nlon + li` (lat index, lon index). This is **not** the NetCDF `source` index.
- **Country raster:** NUTS country polygons (`LEVL_CODE == 0`, else dissolve level-2 by `CNTR_CODE`), rasterized in **ascending polygon area** order so larger countries win at overlaps.
- **CORINE / rasters:** Reprojected to the reference grid with documented resampling (nearest for classes, bilinear for continuous).
- **UWWTD polygons:** Binary coverage (`all_touched=False`). **WWTP points:** Metre buffer in EPSG:3035, union, rasterize; optional Gaussian blur if `scipy` is installed.
- **Within-cell normalization:** Sum of weights = 1 over all fine pixels with valid `cam_cell_id` (≥ 0). If composite proxy sums to 0 in a cell, **uniform** weights over those pixels.

## OSM refinement (optional)

1. Install `osmium-tool` and GDAL `ogr2ogr`.
2. Run [`../Auxiliaries/osm_waste_landuse_extract.py`](../Auxiliaries/osm_waste_landuse_extract.py) on your PBF.
3. Set `paths.osm_waste_gpkg` in `config.yaml`.

A **Greece-only** PBF only refines proxies inside that extract; for Europe-wide refinement use a Europe OSM extract.

## USER MUST CHECK

1. **`CEIP_Waste.xlsx`:** Column names (country, year, sector, pollutant, total), units (Gg vs kg), and sheet name. A pollutant column is **required**.
2. **Sector codes** in the Excel vs [`ceip_sector_codes.yaml`](ceip_sector_codes.yaml) (tokens after stripping non-alphanumerics).
3. **Reference grid vs domain:** A Greece-sized `ref_tif` yields weights only on that window, not the full European CAMS domain.
4. **GHSL SMOD rural codes** in `config.yaml` → `proxy.residual.smod_rural_codes` (legend-specific).
5. **NUTS GeoPackage path** (same convention as Solvents: `Data/geometry/NUTS_RG_20M_2021_3035.gpkg`).

## Incineration exclusion

SNAP incineration codes listed under `exclude_always` in `ceip_sector_codes.yaml` are never included in `E_solid`, `E_ww`, or `E_res`. A future `composite.include_incineration` flag can be wired to relax this after review.
