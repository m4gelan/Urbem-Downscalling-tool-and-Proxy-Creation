# OSM preprocess (sector GeoPackages)

## What this does

Builds multi-layer GeoPackages under `INPUT/Proxy/OSM/` (`waste_layers.gpkg`, `solvents_layers.gpkg`, …) matching `PROXY/config/paths.yaml`, using a unified **osm_engine** (YAML rules for most sectors; Python **plugins** for industry and fugitive).

## Full country build — what you need (including waste)

**There is no separate “waste input file”.** All sectors (waste, solvents, offroad, …) read the **same** OSM **`.osm.pbf`** and the **same** NUTS boundary file. The sector scripts only *filter* that data by tags (landfill, amenities, etc.) and clip to the country mask.

| Input | Role |
|--------|------|
| **Regional / Europe PBF** | Path in `osm_sector_layers.yaml` → `defaults.pbf` (e.g. `INPUT/Preprocess/OSM/_source/europe-latest.osm.pbf`). You must download or place that file yourself (e.g. [Geofabrik](https://download.geofabrik.de/) Europe extract). |
| **NUTS boundaries** | `defaults.nuts_gpkg` (e.g. `INPUT/Proxy/Boundaries/NUTS_RG_20M_2021_3035.gpkg`). Used to get the country polygon and a WGS84 bbox. |
| **`--country EL`** (example) | Two-letter **NUTS `CNTR_CODE`**. Masks and clips outputs to that country only. |
| **osmium-tool** | Recommended: clips the big PBF to the country bbox *before* Python reads it (faster, less RAM). Without it, use `--allow-large-pbf-without-osmium` only for small PBFs or you risk OOM. |

**Flow for one country + one sector (e.g. waste):** download/configure PBF + NUTS → run with `--country` → optional `osmium extract` to bbox → pyosmium walks the (extracted) PBF and keeps objects matching the **waste** rules in `osm_schema.yaml` → write `INPUT/Proxy/OSM/waste_layers.gpkg`.


## Files

| File | Purpose |
|------|--------|
| `osm_sector_layers.yaml` | Run settings: PBF, NUTS, output dir, per-sector options (`with_optional`, `prefilter_tags`, …). |
| `osm_engine/osm_schema.yaml` | Tag / geometry **rules** (nodes, lines, areas), global `min_polygon_area_m2` (default 10 m² in EPSG:3035). |
| `create_osm_sector_packages.py` | CLI that reads `osm_sector_layers.yaml` and runs the right engine or plugin. |
| `create_country_specific_packages.py` | Optional: country bbox extract → roads + landuse/buildings PBFs (not the sector GPKGs). |

## Requirements

- Python: `osmium` (pyosmium), `geopandas`, `shapely`, `pyyaml`
- **osmium-tool** on `PATH` (or pass `--osmium` to the executable) for bbox extract / tags-filter on large PBFs

## Build all sector GPKGs

From the **repository root**:

```bash
python INPUT/Preprocess/OSM/create_osm_sector_packages.py --country EL
```

Use your NUTS `CNTR_CODE` (e.g. `DE`, `FR`). Omit `--country` only if your NUTS file is already a single-country extent.

## One sector

```bash
python INPUT/Preprocess/OSM/create_osm_sector_packages.py --country EL --sector waste
```

Sectors: `waste`, `solvents`, `offroad`, `shipping`, `industry`, `fugitive`.

## Common flags

- `--osmium` — path to `osmium` if not on `PATH`
- `--no-bbox-extract` — read the full PBF in Python (slow, high memory; for small extracts only)
- `--allow-large-pbf-without-osmium` — allow large PBF without osmium-tool (not recommended for continent-scale files)
- `--config` — alternative to `osm_sector_layers.yaml`

## Rules vs plugins

- **Rules** (YAML in `osm_schema.yaml`): waste, solvents, offroad, shipping. Edit predicates there; keep `osm_sector_layers.yaml` for paths and toggles.
- **Plugins** (`osm_engine/plugins/`): `industry_v1`, `fugitive_v1` — complex classification; change the Python if needed.

## Country road / landuse PBFs

```bash
python INPUT/Preprocess/OSM/create_country_specific_packages.py --country EL
```

Outputs next to the script (or `--out-dir`): `OSM_roads_<CC>.osm.pbf`, `OSM_landuse_buildings_<CC>.osm.pbf`.
