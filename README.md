# UrbEM Downscaling Tool and Proxy Creation

This repository builds spatial proxy layers for each GNFR sector of CAMS emissions (area weights and point matching).

## Quick Start: Use Pre-Built Proxies

For countries that have already been processed, sector-specific proxy weights are available for download:

**Google Drive link:** *(add link here)*

After downloading, place the files in the same repository as `proxy/` and `UrbEm_Visualizer/`, under:

```text
INPUT/Proxy_weights/
```

Pre-built proxy weights are currently available for:

- France
- Austria
- Greece
- Belgium
- Spain

If your country is listed above, you can skip proxy generation and go directly to [Run the visualizer](#run-the-visualizer).

## Build proxies yourself

If your country is not covered, or you want to regenerate the proxies, download the full `INPUT/` folder:

**Google Drive link:** *(add link here)*

The folder contains the datasets required by the proxy pipeline. Before running the pipeline, install the Python requirements and prepare the input layers below.

### Prerequisites

- Install the Python requirements:

```bash
pip install -r requirements.txt
```

- Download the OSM Europe GeoPackage (see [INPUT/Proxy/Dataset_sources.md](INPUT/Proxy/Dataset_sources.md)).

### 1. OSM sector packages

Edit `SECTORS`, `SECTORS_ENABLED`, and `COUNTRY` at the top of [INPUT/Preprocess/OSM/create_osm_sector_packages.py](INPUT/Preprocess/OSM/create_osm_sector_packages.py), then run:

```bash
python INPUT/Preprocess/OSM/create_osm_sector_packages.py
```

This writes sector GeoPackages under `INPUT/Proxy/OSM/<Country>/`. Running all sectors can take up to ~12 hours.

### 2. Imperviousness raster (J_Waste only)

Download the Copernicus imperviousness density 100 m raster for your country ([download instructions](INPUT/Proxy/Dataset_sources.md)).

Set `INPUT_DIR` and `OUTPUT_RASTER` in [INPUT/Preprocess/imperviousness.py](INPUT/Preprocess/imperviousness.py), then run:

```bash
python INPUT/Preprocess/imperviousness.py
```

This merges the Copernicus tile zips into a single GeoTIFF.

### Configure and run the pipeline

1. Point the J_Waste config to your imperviousness file in [proxy/config/sector/J_Waste/J_Waste_sector_config.yaml](proxy/config/sector/J_Waste/J_Waste_sector_config.yaml) (`filepaths.Imperviousness.path`).
2. Set `COUNTRY`, `SECTORS_ENABLED`, and other options at the top of [proxy/entry.py](proxy/entry.py).
3. Run:

```bash
python proxy/entry.py
```

Expect roughly one hour for all sectors on a typical country. The generated proxy weights are written under:

```text
INPUT/Proxy_weights/
```

## Run the Visualizer

Once proxy weights are available, start the interface with:

If not done before
```bash
pip install -r requirements.txt
```

Then
```bash
python UrbEm_Visualizer/run.py
```

This launches the local user interface. From there, select or create a run configuration, validate the required inputs, run the downscaling workflow, and inspect the results.