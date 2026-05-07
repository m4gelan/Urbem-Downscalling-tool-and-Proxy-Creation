# GNFR H — Aviation point matching

CAMS-REG **GNFR H** point sources are matched to **OSM aerodrome** candidates with **distance-only** scoring on the same engine as other sectors (`PROXY.core.matching.run_matching`), using facility rows produced by `aviation_matching.build_aviation_facility_candidates`.

## Candidate pool (polygons)

- Source: `paths.yaml` → `osm.aviation` (`aviation_layers.gpkg`), layer `osm_aviation_airport_polygons` (override with `point_matching.aviation_polygon_layer`).
- Keeps `aviation_family` in `{aerodrome, landuse_aerodrome}`.
- **Excludes** `aviation_family == military_airfield` and any feature whose parsed `osm_tags` contain a non-empty **`military`** key.
- **Minimum area 0.5 km²** on **EPSG:3035** geometry (small strips / grass fields dropped).

### Match location inside the polygon

1. If `point_matching.aviation_terminal_apron_layers` lists layers (optionally from `aviation_terminal_apron_gpkg`, defaulting to the main aviation GPKG), features intersecting the aerodrome are unioned, clipped to the aerodrome, and the **centroid** of that intersection is used (with `representative_point()` if the centroid falls outside the clipped part).
2. Otherwise the aerodrome **centroid** is used; if it lies outside the polygon, **`representative_point()`** is used.

Diagnostics keep **polygon centroid** (`polygon_centroid_lon` / `polygon_centroid_lat`) vs **match_location** (`longitude` / `latitude` on the facility row, exported as `facility_longitude` / `facility_latitude` and aliases `match_lon` / `match_lat`).

## Optional aerodrome nodes (fallback)

If `point_matching.aviation_aerodrome_nodes_gpkg` is set (and the file exists), **nodes** tagged `aeroway=aerodrome` are added when they are **not** already covered by a polygon:

- Same **ICAO** as a polygon candidate → skipped.
- Else same **normalized name** and **< 2 km** from a polygon **match_location** → skipped.

Each retained node gets `osm_source = node_buffer`, **match_location** = node coordinates, and `area_km2 = π · (buffer_km)²` with `aviation_node_buffer_m` (default 1000 m). These rows participate in the same greedy matching as polygons.

## Scoring and assignment

- **β_distance = 1**, **β_pollutant = β_activity = 0** (`distance_only: true` in sector YAML).
- Distance score **`q_d = 1 / (1 + d_km / 10)`** (unchanged functional form).
- **Single stage**: no weaker fallback pass; threshold is global **`min_score`** (default 0.4 in `eprtr_scoring.yaml`) applied to the distance-only score.
- **Max distance**: `max_match_distance_km` (10 km for `H_Aviation` via `eprtr_scoring.yaml` / sector override).
- **Nearest shortlist**: 25 (`nearest_candidates`).
- CAMS points sorted by **`co2_total`** (`co2_ff` + `co2_bf` in the NetCDF).
- **One-to-one**: each OSM candidate id and each CAMS point is used at most once; higher CAMS load wins when competing for the same airport (greedy order). **No second-best reassignment** for losers.

## Outputs (under `OUTPUT/Proxy_weights/H_Aviation/`)

The **match table and QA** only need CAMS + OSM; they do **not** require a sector area GeoTIFF.

The **2-band link raster** (`aviation_pointsource.tif`) snaps masses to pixels. By default **`link_grid_match_extent: true`** builds a **small EPSG:3035 grid** around all CAMS + match locations (padding + resolution in YAML). That avoids using full EU **CORINE** as the reference, which would allocate a gigantic array and stall or exhaust RAM.

If you set **`link_grid_match_extent: false`**, the writer falls back to a reference GeoTIFF in this order:

1. `python -m PROXY.main match-points --sector H_Aviation --link-ref-tif <path/to/ref.tif>`
2. `point_matching.link_ref_weights_tif` in this sector YAML
3. The sector **area** weight GeoTIFF if it exists (`output_dir` + `output_filename`)
4. **`paths.yaml` → `proxy_common.corine_tif`**, then **`population_tif`** (refused if larger than ~50M pixels unless you use match-extent).

| Artifact | Description |
|----------|-------------|
| `H_Aviation_point_matches_<year>.csv` | Matches + aviation aliases (`match_lon`, `distance_m`, `stage`, …). |
| `H_Aviation_point_matches_unmatched_<year>.csv` | CAMS H points with no assignment. |
| `H_Aviation_point_matches_qa_<year>.json` / `.csv` | Coverage, distance stats, `node_buffer_match_share`. |
| `H_Aviation_point_matches_<year>_qa.log` | Short text summary. |
| `aviation_pointsource.tif` | Two-band link raster on the area grid (CAMS mass vs match_location mass). |
| `H_Aviation_point_matches_<year>_map.png` | Optional matplotlib diagnostic (lines CAMS → match). |

Run matching (no area proxy required for CSV/QA; link TIF uses CORINE or override if area TIF missing):

```bash
python -m PROXY.main match-points --sector H_Aviation --year 2019 --cams-iso3 GRC
```

Folium context map (requires a **link** GeoTIFF + match CSV; the link uses the same reference grid as above):

```bash
python -m PROXY.main visualize --point-link-only --point-link --sector H_Aviation
```

(`H_Aviation` has no area-context HTML preview in the generic visualize flow; use `--point-link-only` so the CLI does not require `_AREA_PREVIEW_SECTORS`.)

## Area-source downscaling (GNFR H)

CAMS can split GNFR **H** mass between **area** (`source_type_index == 1`) and **point** (`source_type_index == 2`) rows. **Point** allocation is handled by `match-points`; **area** allocation uses `PROXY/sectors/H_Aviation/aviation_area.py` when enabled in `config/sectors/aviation.yaml` under `area_source`.

### Area vs point preflight

For each configured pollutant (CAMS NetCDF variable, or `co2_total` = `co2_ff` + `co2_bf`), the pipeline sums area-row and point-row totals for the chosen **CAMS ISO3** country and GNFR **H**. If **area sum is zero**, that pollutant is **skipped** (all mass is treated as point-only). A CSV summary `aviation_area_summary_<ISO3>_<year>.csv` is written first listing **process** vs **skip** with reasons.

### Binary aerodrome proxy

The fine grid is **EPSG:3035** at **`area_source.resolution_m`** (default **100 m**) over the **NUTS-2 country union** plus padding. The proxy rasterizes **OSM aerodrome polygons** from `build_aviation_aerodrome_pool_gdf` (same rules as point matching: families, military exclusion, ≥0.5 km², apron/terminal centroid, optional **node + buffer** with `osm_source = node_buffer`). Values are **0/1** (uniform inside each polygon). Between CAMS cells, relative allocation is unchanged; **within** each CAMS cell the proxy is normalised to sum to 1 before multiplying by the cell’s area-source mass.

### Background floor

If a CAMS cell has non-zero **area** mass but **no** airport pixels inside that cell on the fine grid, each pixel in the cell receives a tiny weight **`proxy_weight_floor`** (default `1e-6`) before normalisation so the cell mass is preserved and spread almost uniformly. A warning is logged with the cell centre and mass. If the reference window omits a CAMS cell entirely, that mass is reported as lost in QA/metadata.

### Runnable example

Paths resolve from `PROXY/config/paths.yaml` (`emissions.cams_2019_nc`, `osm.aviation`, `proxy_common.nuts_gpkg`). Outputs go to `OUTPUT/Proxy_weights/H_Aviation/` (or `output_dir` in the sector YAML).

```bash
python -m PROXY.main aviation-area --country EL --year 2019 --cams-iso3 GRC
```

Artifacts:

| File | Role |
|------|------|
| `aviation_area_summary_<ISO3>_<year>.csv` | Per-pollutant area/point totals and process/skip |
| `aviation_area_proxy_<ISO3>_<year>.tif` | Binary aerodrome field (float32, GDAL nodata 0) |
| `aviation_area_<pollutant>_<ISO3>_<year>.tif` | Downscaled area emissions (units from CAMS variable metadata) |
| `aviation_area_qa_<ISO3>_<year>.csv` | Per-cell QA (mass, pixel counts, floor flag) |
| `aviation_area_meta_<ISO3>_<year>.json` | Mass balance and roll-ups |
| `aviation_area_aerodrome_pool_<ISO3>_<year>.gpkg` | Clipped aerodrome pool for QA |
| `aviation_area_run_<ISO3>_<year>.log` | Run log (omit with `--no-run-log`) |
