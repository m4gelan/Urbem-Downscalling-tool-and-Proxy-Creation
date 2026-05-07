# PROXY - How to use

This guide explains how to run the `PROXY` package end-to-end: what inputs it
needs, how configuration is organised, which commands are available, where the
outputs land, and what to check when a run fails. It is written for users who
have the codebase on disk and want to reproduce or rebuild a country's proxy
rasters. For the scientific meaning of each sector proxy see
`docs/sector_methodology.md`.

---

## 1. What `PROXY` does

`PROXY` is the downscaling layer of the CAMS-REG-ANT pipeline. Given a set of
reported or modelled emission totals defined on a coarse CAMS grid, it produces
per-sector **spatial weight rasters** (GeoTIFF) on the fine target grid. Every
weight raster is normalised so that the weights of all fine-grid cells inside a
single CAMS cell sum to 1.0 (this is what the downstream redistribution code
expects). The fine-grid-aggregate of any sector's output therefore preserves
the CAMS-cell mass balance exactly.

The package groups its logic into three layers:

1. **Configuration** (`PROXY/config/*.yaml`, `PROXY/config/sectors/*.yaml`,
   `PROXY/config/agriculture/*.yaml`, `PROXY/config/alpha/*.yaml`): paths and
   per-sector parameters.
2. **Core utilities** (`PROXY/core/`): CAMS grid construction, alpha
   computation, raster alignment / normalisation / writing, data discovery,
   dataloaders, matching engine for point sources, structured logging.
3. **Sectors** (`PROXY/sectors/<KEY>/`): one package per sector
   (`A_PublicPower`, `B_Industry`, `C_OtherCombustion`, `D_Fugitive`,
   `E_Solvents`, `G_Shipping`, `I_Offroad`, `J_Waste`, `K_Agriculture`). Each
   sector exposes a `builder.py` (or equivalent) producing the sector's
   multiband or single-band GeoTIFF.

Visualisation helpers live in `PROXY/visualization/` and the CLI entry point is
`PROXY/main.py`.

---

## 2. Prerequisites

### 2.1 Python environment

- Python 3.10+ (the codebase uses `from __future__ import annotations` and PEP
  604 `|` union syntax).
- Required packages (install via `pip install -r requirements.txt` if you
  maintain one; otherwise install on demand):
  - `numpy`, `pandas`, `geopandas`, `fiona`, `shapely`, `pyproj`, `pyogrio`
    (optional backend)
  - `rasterio`, `xarray`, `netCDF4`, `rioxarray`
  - `pyyaml`, `openpyxl`
  - `folium`, `branca`, `jinja2` (visualisation only)
  - `pytest` (tests only)

### 2.2 Input data layout

The paths in `PROXY/config/paths.yaml` are **relative to the repository root**
(the directory that contains the `PROXY/` and `INPUT/` folders). Every entry
must exist before a build is attempted:

```
INPUT/
  Emissions/
    CAMS_REG_ANT_EU_2019.nc            # CAMS fine-grid totals, NetCDF
  Proxy/
    Alpha/
      Reported_Emissions_EU27_2018_2023.xlsx   # reported-emissions workbook
    Boundaries/
      NUTS_RG_20M_2021_3035.gpkg       # NUTS polygons (EPSG:3035)
    CORINE/
      U2018_CLC2018_V2020_20u1_100m.tif
    Population/
      JRC-ESTAT_Census_Population_2021_100m.tif
    EPRTR/
      F1_4_Air_Releases_Facilities.csv
    OSM/
      offroad_layers.gpkg, shipping_layers.gpkg, ...
    ProxySpecific/
      Shipping/vesseldensity_all_2019.tif, Shipping-Lanes-v1.shp
      PublicPower/JRC_OPEN_{UNITS,LINKAGES,PERFORMANCE,TEMPORAL}.csv
      Waste/UWWTD_*.gpkg, greece_imp_100m_mosaic.tif, GHS_SMOD_*.tif
      OtherCombustion/{GAINS,Hotmaps}/..., EUROSTAT/Nrg_d_hhq_*.xlsx
      Agriculture/C21.gpkg, EU_LUCAS_2022.csv, GFED4/..., LUCAS-SOIL-2018-v2/...
```

The discovery helpers (`PROXY/core/dataloaders/discovery.py`) tolerate small
renames of the CAMS and CORINE rasters, but everything else is required
verbatim. If a file is missing, the build of the affected sector aborts with a
clear message (`[build] skip <KEY>: <reason>`); other sectors continue.

### 2.3 Output directories

The CLI writes all outputs under `OUTPUT/Proxy_weights/<sector>/` and, for the
visualisation command, under `OUTPUT/Proxy_visualization/`. Both directories
are created on demand.

---

## 3. Configuration files

### 3.1 Global paths - `PROXY/config/paths.yaml`

Single source of truth for where every input lives. Use this to point the
pipeline at a different data drop without editing Python.

### 3.2 Sector list - `PROXY/config/sectors.yaml`

Enables/disables each sector and points at its sector-level config:

```yaml
sectors:
  - key: K_Agriculture
    type: area
    enabled: true
    config: PROXY/config/sectors/agriculture.yaml
```

Set `enabled: false` to skip a sector globally. `type` is informational
(`area`, `mixed`, `point`).

### 3.3 Per-sector configs - `PROXY/config/sectors/*.yaml`

One YAML per sector. Declares the `output_dir`, `output_filename`, CAMS ISO3
filter, and any sector-specific knobs (e.g. `area_proxy`, pollutant list, OSM
layer selection). A sector YAML always defines at least `output_dir` and
`output_filename`; everything else is sector-specific.

### 3.4 Alpha mapping - `PROXY/config/alpha/*.yaml`

- `mapping_gnfr_to_nfr2.yaml`: GNFR to NFR2 crosswalk used by the alpha
  computer (`PROXY/core/alpha/`).
- `grouped_subsectors.yaml`: collapses NFR2 subsectors into the groups used
  downstream.
- `fallback/defaults.yaml`: cross-country alpha fallbacks (legacy implicit
  defaults, preserved exactly).
- `fallback/<SECTOR>_<ISO2>.yaml`: per-country overrides, one file per sector
  per country. Ships empty so runs are bit-identical to the pre-refactor code.

### 3.5 Agriculture mapping - `PROXY/config/agriculture/*`

- `lu1_lc1_mapping.yaml`: LUCAS LU1/LC1/GRAZING interpretation used by
  `K_Agriculture`. Also hosts the U120 opt-in switch
  (`livestock_housing.u120_mixed_livestock`). U120 is **forest** in the
  shipped default - do not enable it for production runs.
- `emission_factors/*.json`: IPCC/EMEP defaults for each agricultural proxy.

### 3.6 Point-matching scoring - `PROXY/config/matching/eprtr_scoring.yaml`

Used by the `match-points` subcommand for NFR-sector to E-PRTR activity
scoring.

---

## 4. CLI usage

All commands go through `PROXY/main.py`:

```bash
python -m PROXY.main <subcommand> [--config PROXY/config/paths.yaml] [options]
```

`--config` defaults to `PROXY/config/paths.yaml`. Override it to run against a
different data drop.

### 4.1 `build` - generate proxy rasters

```bash
python -m PROXY.main build --sector K_Agriculture --country EL
```

Flags:

| Flag         | Default | Description                                                                                 |
|--------------|---------|---------------------------------------------------------------------------------------------|
| `--sector`   | none    | Sector key (e.g. `A_PublicPower`). Omit to build every sector listed as enabled in `sectors.yaml`. |
| `--country`  | `EL`    | ISO2 country used to clip the NUTS domain. Not all sectors honour this (global sectors ignore it). |
| `--config`   | `PROXY/config/paths.yaml` | Override path config.                                                         |

The build loop:

1. Loads `paths.yaml` and resolves CAMS + CORINE via the discovery helpers.
2. Iterates `sectors.yaml` entries with `enabled: true` (or just the one
   selected via `--sector`).
3. Imports `PROXY.sectors.<KEY>.builder` and calls `builder.build(path_cfg,
   sector_cfg, country)`.
4. Prints a per-sector line and a final summary:
   ```
   [build] K_Agriculture -> OUTPUT/Proxy_weights/K_Agriculture/agriculture_areasource.tif
   [build] completed built=1 skipped=0
   ```

A sector that raises is **skipped**, not fatal - other sectors keep running.
Read the last line (`built=N skipped=M`) and check logs for the sectors that
skipped.

### 4.2 `alpha` - compute national alpha factors

```bash
python -m PROXY.main alpha --country EL --pollutant NOx
```

Reads the reported-emissions workbook, applies the GNFR->NFR2 mapping and
grouped-subsector aggregation, and writes alpha tables under
`OUTPUT/Proxy_weights/_alpha/`. Omit `--pollutant` to cover every pollutant
present for that country.

### 4.3 `match-points` - CAMS-to-facility matching

```bash
python -m PROXY.main match-points --sector B_Industry --year 2019 --cams-iso3 GRC \
    --pollutant NOX --max-points 50000
```

Matches CAMS grid cells to E-PRTR facilities using the scoring config. Output
goes to `OUTPUT/Proxy_weights/<sector>/`. `--cams-iso3` uses CAMS's 3-letter
country code (e.g. `GRC`, not `EL`).

### 4.4 `visualize` - render HTML previews

```bash
python -m PROXY.main visualize --sector K_Agriculture --country EL \
    --weight-display global_log
```

Options:

- `--output`: folder (default `OUTPUT/Proxy_visualization`).
- `--out-html`: specific output path (default `<output>/<sector>_area_context_map.html`).
- `--weight-tif`: override the raster to preview (default: sector's
  `output_filename`).
- `--pad-deg`, `--max-width`, `--max-height`: extent and overlay sizing.
- `--weight-display`: `global_log` (legacy look, colour is log10 of the raw
  weight) or `per_cell` (each CAMS cell rescaled 0-1).
- `--no-cams`: skip loading the CAMS NetCDF (useful when only raster checks
  are needed).

The sector-specific renderer is imported lazily; adding a new sector requires
adding an `elif` in `main._visualize_cmd` *and* a writer function under
`PROXY/visualization/`.

### 4.5 `validate` - quality checks

```bash
python -m PROXY.main validate --sector K_Agriculture --strict
```

Thin wrapper today; will grow as more QA checks are added.

---

## 5. Typical recipes

### 5.1 Build every sector for Greece

```bash
python -m PROXY.main build --country EL
```

### 5.2 Build a single sector and preview it

```bash
python -m PROXY.main build --sector D_Fugitive --country EL
python -m PROXY.main visualize --sector D_Fugitive --country EL
```

### 5.3 Dry-run with discovery diagnostics

The discovery helpers print where they looked. Run:

```bash
python -m PROXY.main build --sector A_PublicPower --country EL 2>&1 | head -40
```

and check the first lines for messages like
`[discover] CORINE resolved to INPUT/Proxy/CORINE/...`.

### 5.4 Run the agriculture regression tests

```bash
python -m pytest PROXY/sectors/K_Agriculture/tests -x --no-header -q
```

29 tests should pass. These pin the U120 opt-out default, the JSON-backed
emission factors, and the LC1/LU1 mapping integrity.

### 5.5 Opt-in to the U120 research scoring

(Not recommended for production.) Edit
`PROXY/config/agriculture/lu1_lc1_mapping.yaml`:

```yaml
livestock_housing:
  u120_mixed_livestock:
    enabled: true
    score: 0.55
```

A `WARNING` line with `U120 reinterpretation ACTIVE` will appear in the build
logs. Revert by setting `enabled: false` (or by restoring `score: null`).

---

## 6. Troubleshooting

### 6.1 `[build] skip <KEY>: builder module not found`

`PROXY/sectors/<KEY>/builder.py` does not exist. This is expected only for
keys that do not ship a builder. Check that the `<KEY>` matches the sector
list in `sectors.yaml`.

### 6.2 `[build] skip <KEY>: <missing file>`

A required input was not found. Check:

1. `PROXY/config/paths.yaml` points to an existing absolute or root-relative
   path.
2. The file is readable (Windows: no long-path or permission issue).
3. For CAMS and CORINE, the discovery helper fell back correctly (a second
   log line usually shows what was tried).

### 6.3 Output weights do not sum to 1 per CAMS cell

Every sector runs `normalize_within_cams_cells` before writing. If the check
in `validate_weight_sums` warns about deviations:

- Confirm that the weight raster has the expected CRS and resolution (the
  grid-alignment code in `PROXY/core/raster/align.py` logs the target grid).
- A CAMS cell with zero total proxy mass falls back to a uniform distribution
  (logged at `INFO`) - expected, not a bug.
- A CAMS cell with some NaN pixels can drift above 1 if the NaN mask is not
  excluded; re-check the sector's `valid_mask` construction.

### 6.4 `ModuleNotFoundError: Waste.j_waste_weights` or `Shipping.shipping_areasource`

The refactor migrated away from those external modules. Update your import to
the new core locations:

| Old import                                            | New import                                                |
|-------------------------------------------------------|-----------------------------------------------------------|
| `Waste.j_waste_weights.cams_grid.build_cam_cell_id`   | `PROXY.core.cams.grid.build_cam_cell_id`                  |
| `Waste.j_waste_weights.io_utils.warp_raster_to_ref`   | `PROXY.core.raster.warp_raster_to_ref`                    |
| `Waste.j_waste_weights.normalization.normalize_within_cams_cells` | `PROXY.core.raster.normalize_within_cams_cells` |
| `Waste.j_waste_weights.country_raster.rasterize_country_ids` | `PROXY.core.raster.rasterize_country_ids`          |
| `Shipping.shipping_areasource.resolve_corine_tif`     | `PROXY.core.corine_masks.resolve_corine_tif`           |

The old packages are not needed anywhere in `PROXY/` any more.

### 6.5 Visualisation renders but shows no overlay

- The `--weight-tif` defaults to `sector_cfg["output_dir"] /
  sector_cfg["output_filename"]`. If you renamed the build output, pass
  `--weight-tif` explicitly.
- `--no-cams` disables the CAMS-cell grid. Remove it to see the grid.
- If the overlay is empty, confirm the weight raster has non-zero data inside
  the map's bounding box (`--pad-deg` controls the pad).

### 6.6 Pytest cannot import `PROXY`

Run `pytest` from the repository root, not from `PROXY/`. The codebase relies
on the top-level package path (`PROXY.sectors.K_Agriculture.tests.*`).

---

## 7. Where to look next

- Scientific methodology and per-sector formulas: `docs/sector_methodology.md`.
- Refactor history, bug fixes, drift flags: `docs/refactor_notes.md`.
- Agriculture class interpretation (LC1/LU1/GRAZING, CORINE bands):
  `docs/agriculture_class_mapping.md`.
- Alpha-fallback YAML conventions: comments at the top of
  `PROXY/config/alpha/fallback/defaults.yaml`.
