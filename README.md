# UrbEM Downscaling Tool and Proxy Creation

Welcome to the V2 UrbEm dowscaling tool. This repository builds air-pollution emission maps for European cities and regions at **100 m** or **1 km** resolution. It uses the **CAMS-REG-ANT** inventory (emissions by GNFR sector on a European grid) and spatial **proxy weights** to distribute those totals onto a finer grid.

This work was developed for the **Metronome** city project by Léo Pillac-Mage.

---

## How the pipeline fits together

1. **INPUT/** — raw datasets (CAMS emissions, OSM extracts, CORINE, population, etc.) and preprocessing scripts.
2. **proxy/** — builds sector proxy weights (GeoTIFFs) from those inputs.
3. **UrbEm_Visualizer/** — applies proxy weights to CAMS totals for a chosen domain and writes downscaled emission grids (100 m or 1 km, selectable in the UI).
4. **Transform_for_citychem/** — converts UrbEm outputs from GNFR folders into **SNAP** CSV files for CityChem (area, point, and road line sources).

The **Urbem_V1/** folder holds the original **R** downscaling scripts (reference / legacy). The current workflow uses the Python stack above.

Typical flow: get proxy weights → run the visualizer → optionally transform for CityChem.

---

## Repository layout

```text
PDM_local/
├── INPUT/
│   ├── Emissions/              # CAMS-REG NetCDF (e.g. year 2019, 2021)
│   ├── Proxy/                  # Reference datasets + OSM sector packages
│   ├── Proxy_weights/          # Proxy GeoTIFFs (download or build with proxy/)
│   └── Preprocess/             # OSM packages, imperviousness, etc.
├── proxy/
│   ├── entry.py                # Main proxy build script (edit settings at top)
│   └── config/                 # Sector configs and file paths
├── UrbEm_Visualizer/
│   ├── run.py                  # Desktop UI or headless CLI
│   └── config/run/             # Run configs (auto-saved when you run from the UI)
├── Urbem_V1/                   # Legacy R downscaling (CAMS v8.1, UECT / CityChem CSV)
│   ├── 1_UrbEm_pointsources_CAMS8.1.R
│   ├── 2_UrbEm_areasources_CAMS8.1.R
│   ├── 3_UrbEm_linesources_CAMS8.1.R
│   └── *.R                     # Helper scripts (CAMS read, proxies, OSM roads)
├── Transform_for_citychem/
│   ├── transform.py            # GNFR → SNAP export for CityChem
│   └── config.yaml             # City domain, paths, SNAP mapping
└── Output/                     # Created when you run pipelines (not in git)
    ├── UrbEm/<run_name>/       # New downscaling results (UrbEm_Visualizer)
    └── CityChem/<city>/        # CityChem CSVs (Transform_for_citychem)
```

For a full list of proxy input datasets and download URLs, see [INPUT/Proxy/Dataset_sources.md](INPUT/Proxy/Dataset_sources.md).

---

## CAMS emissions data

**Jeroen Kuenen (TNO)** — jeroen.kuenen@tno.nl — is the contact for **CAMS-REG** inventory access.

Through the project API (coming soon), anyone can download CAMS for the supported reference years:

```text
GET /api/v1/cams?year=2019
GET /api/v1/cams?year=2021
```

Place the NetCDF files under `INPUT/Emissions/` (see [UrbEm_Visualizer/config/expected_inputs_filepaths.yaml](UrbEm_Visualizer/config/expected_inputs_filepaths.yaml)).

All proxy weights and downscaling workflows in this repo were built for **CAMS 2019**, and the same pipeline can be run for **CAMS 2021**. For **other CAMS years**, contact Jeroen Kuenen for **FTP access** from TNO — those files are not served through the API.

---

## Quick start: use pre-built proxy weights

For countries that have already been processed, sector proxy weights can be fetched through the project download API (coming soon):

```text
GET /api/v1/proxy-weights?country=<Country>&year=<CAMS_YEAR>
```

After download, place the GeoTIFFs under:

```text
INPUT/Proxy_weights/
```

Pre-built weights are currently available for:

- France, Austria, Greece, Belgium, Spain, Italy, Germany, Switzerland

These were built for **CAMS 2019**; the same countries can be processed for **CAMS 2021** by rebuilding proxies with `CAMS_YEAR = 2021` in [proxy/entry.py](proxy/entry.py).

If your country is listed above, skip proxy generation and go to [Run the visualizer](#run-the-visualizer).

---

## Build proxy weights yourself

Use this path when your country is not covered, or you want to regenerate proxies for a different CAMS year.

Input datasets can also be retrieved via the project API (coming soon):

```text
GET /api/v1/input-datasets?bundle=proxy
```

Until then, see [INPUT/Proxy/Dataset_sources.md](INPUT/Proxy/Dataset_sources.md) for manual download links.

### Prerequisites

```bash
pip install -r requirements.txt
```

You also need the **OSM Europe GeoPackage** (see Dataset_sources.md). For large countries, install **osmium-tool** on your PATH.

### 1. OSM sector packages

Edit `SECTORS`, `SECTORS_ENABLED`, and `COUNTRY` at the top of [INPUT/Preprocess/OSM/create_osm_sector_packages.py](INPUT/Preprocess/OSM/create_osm_sector_packages.py), then run:

```bash
python INPUT/Preprocess/OSM/create_osm_sector_packages.py
```

This writes sector GeoPackages under `INPUT/Proxy/OSM/<Country>/`. Running all sectors can take several hours.

### 2. Imperviousness raster (J_Waste only)

Download the Copernicus imperviousness density 100 m raster for your country ([instructions](INPUT/Proxy/Dataset_sources.md)).

Set `INPUT_DIR` and `OUTPUT_RASTER` in [INPUT/Preprocess/imperviousness.py](INPUT/Preprocess/imperviousness.py), then run:

```bash
python INPUT/Preprocess/imperviousness.py
```

### 3. Configure and run the proxy pipeline

Edit the settings at the top of [proxy/entry.py](proxy/entry.py):

| Setting | Purpose |
|---------|---------|
| `COUNTRIES` | One or more countries to process |
| `SECTORS_ENABLED` | GNFR sectors to build (A–K) |
| `CAMS_YEAR` | `2019` or `2021` (must match your CAMS NetCDF) |
| `AREA_WEIGHTS` / `POINT_MATCHING` | Which proxy products to generate |
| `LOG_LEVEL` | `INFO` for normal runs; `DEBUG` for debug maps |
| `CITY` + `MAP_TYPE` | Debug maps only (`INTERACTIVE` → HTML, `FIXED_IMAGE` → PNG) |

Then run:

```bash
python proxy/entry.py
```

Expect roughly one hour for all sectors on a typical country. Outputs are written to:

```text
INPUT/Proxy_weights/<Sector>/
```

File naming follows `{Sector}_{Country}_area_weights_{year}.tif` (and point-source variants where applicable). Debug city bounds are defined in [proxy/visualizers/bouding_boxes.yaml](proxy/visualizers/bouding_boxes.yaml).

---

## Run the visualizer

Once proxy weights and CAMS emissions are in place:

```bash
pip install -r requirements.txt
python UrbEm_Visualizer/run.py
```

This opens a local desktop app (PyWebView + Flask). From there you can:

- choose country, CAMS year, pollutants, and domain on the map
- validate that required proxy and CAMS files exist
- run the downscaling workflow at **100 m** or **1 km** resolution
- inspect results on the map and in the statistics panel

**You do not need to write a YAML config by hand.** The app builds the run configuration from your choices (country, validated inputs, domain, pollutants, output resolution) and saves it automatically to `UrbEm_Visualizer/config/run/` when you start a run.

Downscaled outputs are written to **`Output/UrbEm/<run_name>/`**. This folder is **created automatically** when you run the pipeline (one subfolder per run, named from your config). It is not part of the repository — see `.gitignore`.

### Headless run (single sector)

```bash
python UrbEm_Visualizer/run.py --no-ui --config UrbEm_Visualizer/config/run/Athens_2019.yaml --sector C_OtherCombustion
```

Omit `--sector` to run all sectors defined in the config.

### Example run configs (reference / headless only)

The UI writes configs for you. The files below are **saved examples** from past runs — useful mainly for `--no-ui` CLI runs or inspection:

| Config | Country | Notes |
|--------|---------|-------|
| [Athens_2019.yaml](UrbEm_Visualizer/config/run/Athens_2019.yaml) | Greece | 1 km output |
| [Ioannina_2019.yaml](UrbEm_Visualizer/config/run/Ioannina_2019.yaml) | Greece | 100 m, city domain |
| [Kozani_2019.yaml](UrbEm_Visualizer/config/run/Kozani_2019.yaml) | Greece | 100 m, city domain |

Expected input layout and sector modes (area-only vs point vs both) are documented in [UrbEm_Visualizer/config/expected_inputs_filepaths.yaml](UrbEm_Visualizer/config/expected_inputs_filepaths.yaml).

---

## UrbEm V1 (R scripts)

The [Urbem_V1/](Urbem_V1/) folder contains the **original UrbEm downscaling workflow in R**, updated for **CAMS-REG-ANT v8.1**. These scripts are kept for reference and comparison with the new Python tool. Original authors: M.O.P. Ramacher & A. Kakouri (2021); CAMS v8.1 update (2025).

Run in this order:

| Script | Role |
|--------|------|
| [1_UrbEm_pointsources_CAMS8.1.R](Urbem_V1/1_UrbEm_pointsources_CAMS8.1.R) | Point sources — distribute CAMS grid emissions to RI-URBANS locations; unmatched mass goes back to area sources |
| [2_UrbEm_areasources_CAMS8.1.R](Urbem_V1/2_UrbEm_areasources_CAMS8.1.R) | Area sources — downscale GNFR sectors with proxy rasters; GNFR → SNAP; UECT area CSV |
| [3_UrbEm_linesources_CAMS8.1.R](Urbem_V1/3_UrbEm_linesources_CAMS8.1.R) | Line sources — road traffic (F1–F4) from area grids onto OSM roads; UECT line CSV |

Helper scripts in the same folder:

- `prepare_cams_v8.1_TNOftp_emissions.R` — read and rasterize CAMS v8.1 NetCDF by sector
- `proxy_preparation.R` — normalize proxies within coarse CAMS cells and apply weights
- `areasources_to_osm_linesources.R` — allocate area road emissions to OSM line geometry

Each main script has an **INPUT** block at the top (paths, urban domain, site name, output folder). Set `setwd()` to the repository root. Helper scripts live in `Urbem_V1/` — update any `source("./Urbem_V1/UrbEm_v1.0_Python_script/...")` lines to `./Urbem_V1/...` if needed.

Outputs are **UECT-formatted CSV** files for EPISODE-CityChem. Requires R packages such as `raster`, `sf`, `ncdf4`, `reshape2`, and for roads `osmdata`, `lwgeom`.

For new work, use **UrbEm_Visualizer** and **Transform_for_citychem** instead.

---

## Transform for CityChem

The scripts under `Transform_for_citychem/` convert UrbEm downscaling results into **SNAP** source files for CityChem:

- `area_source_<City>.csv`
- `point_source_<City>.csv`
- `line_source_<City>_F*.csv` (road categories F1–F4)

Edit [Transform_for_citychem/config.yaml](Transform_for_citychem/config.yaml) (`Input_folder`, `Output_folder`, `domain`, `EPSG`, `Grid_step_m`, `Roads_lines`, `SNAP_TO_GNFR`). Set `Input_folder` to your downscaling run under `Output/UrbEm/<run_name>/`, then run:

```bash
python Transform_for_citychem/transform.py
```

GNFR → SNAP mapping used in the config:

```yaml
SNAP_TO_GNFR:
  SNAP_1: A_PublicPower
  SNAP_2: C_OtherCombustion
  SNAP_3: B_Industry
  SNAP_5: D_Fugitive
  SNAP_6: E_Solvents
  SNAP_7: F_Roads
  SNAP_8:
    - G_Shipping
    - H_Aviation
    - I_Offroad
  SNAP_9: J_Waste
  SNAP_10: K_Agriculture
```

---

## Credits and contact

| Role | Detail |
|------|--------|
| Development | Léo Pillac-Mage |
| Project | Metronome city project/Master's thesis at EPFL |
| CAMS data | Copernicus Atmosphere Monitoring Service (CAMS-REG) |
| CAMS access (API 2019/2021; FTP for other years) | jeroen.kuenen@tno.nl (TNO) |
| Proxy input documentation | [INPUT/Proxy/Dataset_sources.md](INPUT/Proxy/Dataset_sources.md) |

Supported CAMS years in this repo: **2019** and **2021** (via API when live). For other years, request FTP access from TNO. To use a different year end-to-end, set the matching `CAMS_YEAR` in `proxy/entry.py` and select the same year in the visualizer.
