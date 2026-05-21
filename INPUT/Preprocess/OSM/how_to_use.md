# OSM preprocess (sector GeoPackages)

## What this does

Builds multi-layer GeoPackages (e.g. `waste_layers.gpkg`, `industry_layers.gpkg`) for **PROXY_V2** sector pipelines. Each file is read from `filepaths.OSM.path` in the matching `PROXY_V2/config/sector/*/` config.

All sectors share one **osm_engine**: YAML **rules** (`mode: rules`) or ordered **classify** rules (`mode: classify` for industry). There are no per-sector Python plugins.

## Inputs (same for every sector)

| Input | Where configured |
|--------|------------------|
| **OSM `.osm.pbf`** | `osm_sector_layers.yaml` → `defaults.pbf` (e.g. `INPUT/Preprocess/OSM/_source/europe-latest.osm.pbf`). Download from [Geofabrik](https://download.geofabrik.de/) or similar. |
| **NUTS boundaries** | `defaults.nuts_gpkg` (e.g. `INPUT/Proxy/Boundaries/NUTS_RG_20M_2021_3035.gpkg`). |
| **Country** | `COUNTRY` in `create_osm_sector_packages.py` (human name, e.g. `Greece` → NUTS `CNTR_CODE` `EL`). |
| **Output folder** | `OUTPUT_DIR` in `create_osm_sector_packages.py` (overrides `defaults.output_dir` in YAML). |
| **osmium-tool** | On `PATH`, or set `OSMIUM_EXE`. Used for country bbox extract and optional tags-filter before pyosmium. |

**Flow:** load NUTS mask for country → **one shared** bbox extract per run → optional tags-filter per sector → pyosmium scan → clip/dedupe/min-area → write `{sector}_layers.gpkg` under `OUTPUT_DIR`.

## How to run

Edit the top of [`create_osm_sector_packages.py`](create_osm_sector_packages.py), then from the **repository root**:

```bash
python INPUT/Preprocess/OSM/create_osm_sector_packages.py
```

### User settings (top of `create_osm_sector_packages.py`)

| Constant | Purpose |
|----------|---------|
| `SECTORS` | All sector ids known to the engine |
| `SECTORS_ENABLED` | Subset to build in this run |
| `COUNTRY` | Country name (must be in `COUNTRY_TO_CNTR`) |
| `OUTPUT_DIR` | Folder for `*_layers.gpkg` files |
| `LOG_LEVEL` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `NO_BBOX_EXTRACT` | If `True`, skip osmium bbox extract (slow / high RAM on large PBF) |
| `ALLOW_LARGE_PBF_WITHOUT_OSMIUM` | Allow continent-scale PBF without osmium-tool |
| `OSMIUM_EXE` | Path to `osmium` executable, or `None` for `PATH` |

Example:

```python
SECTORS_ENABLED = ["waste", "fugitive"]
COUNTRY = "Greece"
OUTPUT_DIR = "INPUT/Proxy/OSM"
LOG_LEVEL = "INFO"
```

## Sectors

`waste`, `solvents`, `offroad`, `shipping`, `industry`, `fugitive`, `aviation`, `agricultural`

| Sector | Engine mode | Output (default names) |
|--------|-------------|-------------------------|
| waste | rules | `osm_waste_polygons`, `osm_waste_points` |
| solvents | rules | roads, landuse, buildings, … |
| offroad | rules | rail / pipeline layers |
| shipping | rules | `osm_shipping_high`, `osm_shipping_medium` |
| industry | classify | `osm_industrial_{polygons,lines,points}`, column `industrial_layer` |
| fugitive | rules | `osm_energy_polygons`, column `tag_match_families` |
| aviation | rules | `osm_aviation_airport_polygons` |
| agricultural | rules | `agricultural_farmyard`, `agricultural_points_buffered` |

## Configuration files

| File | Purpose |
|------|---------|
| [`osm_sector_layers.yaml`](osm_sector_layers.yaml) | PBF/NUTS paths, `min_polygon_area_m2`, per-sector flags (`prefilter_tags`, `with_optional`, `include_wastewater_plant`, …). |
| [`osm_engine/osm_schema.yaml`](osm_engine/osm_schema.yaml) | Tag rules, layer order, `augment` columns, `osmium_tag_filters`, industry/fugitive semantics. |
| [`osm_engine/industry_classify_rules.yaml`](osm_engine/industry_classify_rules.yaml) | Ordered classify rules for industry (loaded into schema at runtime). |

Change **what** gets extracted: edit `osm_schema.yaml` (and industry classify rules). Change **paths and run toggles**: edit `osm_sector_layers.yaml` and/or the entry script constants.

## Requirements

- Python: `osmium` (pyosmium), `geopandas`, `shapely`, `pyyaml`
- **osmium-tool** strongly recommended for Europe-scale PBFs

## Performance notes

- Bbox extract runs **once per run** and is reused for all sectors in `SECTORS_ENABLED`.
- Sectors with `osmium_tag_filters` in `osm_schema.yaml` should use `prefilter_tags: true` (default in `osm_sector_layers.yaml`) so pyosmium reads a small PBF, not the full country extract. **Solvents without prefilter can take 100+ minutes and gigabytes of RAM.**
- Default `pyosmium_idx: flex_mem` (portable). After `tags-filter`, PBFs are small enough for `flex_mem` on typical RAM.
- `LOG_LEVEL=DEBUG` does **not** store full `osm_tags` unless you set `store_osm_tags: true` on a sector.
- Run fewer sectors per invocation (`SECTORS_ENABLED`) if memory is tight.
- Prefer a **country extract** PBF (e.g. Geofabrik `austria-latest.osm.pbf`) instead of filtering all of Europe to one country.

## Logging

Messages use the `[osm]` prefix, e.g. `[osm][waste] parse done kept=8691 (27m39s)`. Increase detail with `LOG_LEVEL = "DEBUG"`.

## Rules vs classify

- **`mode: rules`** — `RulesCollector` matches `rules.nodes` / `lines` / `areas` in `osm_schema.yaml`; optional `augment.columns` add derived fields (`waste_family`, `tag_match_families`, …).
- **`mode: classify`** — `ClassifyCollector` applies first-match `classify_rules` (industry only today); output column `industrial_layer`.

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| osmium OOM on extract | Retry (engine retries without progress bar); use a smaller regional PBF; ensure enough RAM |
| Fugitive / industry empty after prefilter | `prefilter_tags: false` for that sector in `osm_sector_layers.yaml` |
| Unknown `COUNTRY` | Add mapping in `COUNTRY_TO_CNTR` in the entry script |
| PROXY_V2 cannot find GPKG | Point `filepaths.OSM.path` at the same folder as `OUTPUT_DIR` |
