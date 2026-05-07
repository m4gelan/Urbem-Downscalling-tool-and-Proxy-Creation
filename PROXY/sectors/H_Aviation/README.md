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

| Artifact | Description |
|----------|-------------|
| `H_Aviation_point_matches_<year>.csv` | Matches + aviation aliases (`match_lon`, `distance_m`, `stage`, …). |
| `H_Aviation_point_matches_unmatched_<year>.csv` | CAMS H points with no assignment. |
| `H_Aviation_point_matches_qa_<year>.json` / `.csv` | Coverage, distance stats, `node_buffer_match_share`. |
| `H_Aviation_point_matches_<year>_qa.log` | Short text summary. |
| `aviation_pointsource.tif` | Two-band link raster on the area grid (CAMS mass vs match_location mass). |
| `H_Aviation_point_matches_<year>_map.png` | Optional matplotlib diagnostic (lines CAMS → match). |

Run matching after the **area** reference GeoTIFF exists:

```bash
python -m PROXY.main match-points --sector H_Aviation --year 2019 --cams-iso3 GRC
```

Folium context map (requires `aviation_areasource.tif`, match CSV, and `aviation_pointsource.tif`):

```bash
python -m PROXY.main visualize --point-link-only --point-link --sector H_Aviation
```

(`H_Aviation` has no area-context HTML preview in the generic visualize flow; use `--point-link-only` so the CLI does not require `_AREA_PREVIEW_SECTORS`.)
