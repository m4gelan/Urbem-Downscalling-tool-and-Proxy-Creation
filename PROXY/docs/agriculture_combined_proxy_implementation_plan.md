# K_Agriculture: implementation plan for combined spatial proxies

This document tracks the refactor from the legacy **NUTS2×CLC tabular `rho` → single CORINE-rasterised score per pollutant** workflow to a **multi-family raster stack → CAMS-cell normalisation** workflow aligned with GNFR J / C-style combined proxies.

## Step 0 --- OSM agriculture GeoPackage (done: contract + path wiring)

**Path (resolved in builds):** `paths.yaml` → `osm.agriculture` → default file `INPUT/Proxy/OSM/agricultural_layers.gpkg`.

**Machine-readable contract:** `PROXY/config/agriculture/osm_agriculture_layers.yaml` defines:

- mandatory **layer name** `agricultural_farmyard` (polygons / multipolygons);
- optional `agricultural_points_buffered` (polygons from buffered OSM points);
- expected CRS, rasterization flags, and a **preprocess tag reference** (human documentation for the PBF→GPKG maintainer).

**Runtime wiring:** `merge_k_agriculture_pipeline_cfg` stores the resolved absolute path under `cfg["paths"]["inputs"]["agriculture_osm_gpkg"]` when `osm.agriculture` is set, and logs it. Rasterization of \(H_j\) is implemented in a later step (`signals/housing_pasture.py`).

**Air pollutants only:** `alpha.config.yaml` and `config/ceip/alpha/alpha_methods.yaml` no longer list CH\(_4\) or CO\(_2\) for K\_Agriculture. Default raster pollutant fallback in `pipeline.py` is `NH3`, `NOx`, `NMVOC`, `CO`, `PM10`, `PM2.5`.

## Target architecture

1. **Preprocess** (outside `run_k_agriculture_pipeline` or as explicit first stage): OSM farm extract (output = `agricultural_layers.gpkg`); Köppen raster from KMZ; optional GFED refresh.
2. **Per-family rasters** `P_g` on the CORINE reference window (same `reference_window_profile` as today).
3. **CAMS aggregation**: for each CAMS area cell, compute `S_{p,j} = Σ_g α_{g,p,c} P_{g,j}` then `W_{p,j} = S_{p,j} / Σ_{k∈C} S_{p,k}` (reuse patterns from `PROXY.core.cams` / other sectors where possible).
4. **Configuration**: new YAML (e.g. `PROXY/config/agriculture/agriculture_combined_rules.yaml`) for `f_s^farm`, `LSU_s`, Köppen wet/dry map, grazing amplification (`κ`, `ε`, `w_max`), minor CORINE class list, OSM/LUCAS weights, CORINE 231/243 weights.

## Files likely to become obsolete or superseded

| Area | Item | Reason |
|------|------|--------|
| Tabular scoring | `tabular/pipeline.py` `PROCESS_REGISTRY` + `compute_pollutant_score` as **sole** path | Replaced by raster `P_g` stack + combined score; tabular CSV may shrink to NUTS2-side tables only (λ_n, γ(n), EF_eff) or be removed from the hot path. |
| Raster output | `rasterize_kl.py` reading `weights_long.csv` `w_p` per pollutant from single merged score | Must write multiband weights from combined `S_{p,j}` or multistage intermediate GeoTIFFs. |
| Process modules | `enteric_grazing.py` as **standalone** CH4 proxy | Enteric spatial pattern is folded into **Family 1** (λ split) and **Family 3** (grazing pasture); CH4 α must be remapped to families (or a residual “enteric-only” band kept with explicit decision). |
| Process modules | `livestock_housing.py` as **sole** NH3/PM housing proxy | Superseded by **Family 1** OSM+LUCAS+CORINE; housing tiers may be mined for fallback scores inside `H_j` only. |
| Crop / burning | `agricultural_soils.py` (NMVOC-only LC1 table) | Superseded by **Family 5** with PM + NMVOC + Köppen γ + five-group map ψ. |
| Crop / burning | `biomass_burning.py` LUCAS residue split + GFED | **Family 7** uses GFED-only spatial proxy per agreed design; residue/LUCAS logic **removed from spatial** unless retained for validation or for a different NFR. |
| Config coupling | `alpha.config.yaml` `process_id` tokens (`enteric_grazing`, `livestock_housing`, …) | Must align with **seven families** (or expanded `G`) and CEIP subgroup keys when α is driven from reported data. |

**Not obsolete (reuse / extend):**

- `tabular/class_extent.py`, `tabular/zonal.py` — useful for NUTS2 zonal stats (Köppen majority, C21 merge).
- `source_relevance/census_intensity.py` / `aggregate_c21_heads_by_nuts2` — **λ_n** and LSU weights.
- `source_relevance/lucas_points.py`, `lucas_survey.py`, `common.py` — LUCAS I/O and `(NUTS_ID, CLC_CODE)` aggregation for families 2–5.
- `source_relevance/manure.py`, `fertilized_land.py` — science for μ^{(2)}, μ^{(4)}; may be refactored into `signals/manure_application.py` etc.
- `k_config.py`, `builder.py`, `pipeline.py` — orchestration shell stays; internals swap.
- `tabular/emission_factors.py` — extend loaders for `f_s^farm`, `LSU_s`, Köppen map, EMEP EF tables.
- `config/agriculture/emission_factors.yaml` — add sections; keep `c21_census_field_map`.
- `visualization/agriculture_area_map.py` — update to preview **per-family** bands and combined weights.

## New / heavily rewritten components

### INPUT preprocess (`INPUT/Preprocess/OSM` or mirror under `PROXY/tools`)

- Script or Makefile target: build `INPUT/Proxy/OSM/agricultural_layers.gpkg` from OSM PBF using tags listed in `osm_agriculture_layers.yaml` under `preprocess_osm_tag_reference`.
- Document CRS, buffer radii, and deduplication rules.

### Köppen

- One-off or scripted: unzip KMZ → raster aligned to project CRS; build integer code → Köppen symbol table; join to **frozen** `wet` / `dry` YAML.
- Runtime: zonal majority (or area-weighted vote) over NUTS2 polygons → `γ(n)` table consumed by Family 5.

### `PROXY/sectors/K_Agriculture/` package layout (proposal)

```
K_Agriculture/
  signals/                 # NEW: raster builders returning ndarray same shape as ref grid
    housing_pasture.py     # Families 1: H, P, λ merge
    manure_application.py  # thin wrapper calling existing manure μ + rasterise
    grazing_pasture.py     # Family 3: CORINE polygons + LUCAS amplification
    synthetic_n.py         # Family 4
    crop_operations.py     # Family 5: EF_eff × arable mask
    minor_soil.py          # Family 6
    field_burning.py       # Family 7: GFED warp
  combine.py               # NEW: S_p,j, W_p,j given P_g stack and α
  rasterize_kl.py          # REWRITE: multiband output from combine
  tabular/pipeline.py      # NARROW: optional diagnostic CSVs; or split into signals-only
```

### `PROXY/config/paths.yaml`

- `osm.agriculture` → `INPUT/Proxy/OSM/agricultural_layers.gpkg` (farm infrastructure for Family~1).
- Later: `koppen_raster_tif` under `proxy_specific.agriculture` or `osm` block as needed.

### CEIP / alpha engine

- If α is computed from reported subgroups: new profile YAML mapping **NFR / CEIP rows → family index `g`** (similar to `C_OtherCombustion_rules.yaml`).
- If α stays frozen: update `alpha.config.yaml` pollutant rows to reference `family_1` … `family_7` instead of legacy `process_id` names.

### Tests

- `PROXY/sectors/K_Agriculture/tests/`: golden small window (few CAMS cells) with synthetic CORINE + fake P_g to assert `Σ_{j∈C} W_{p,j} = 1`.
- Point-in-polygon tests for grazing amplification counts `n_m`.

## Migration phases (recommended)

1. **Dual-write phase**: compute new `P_g` rasters and write to `OUTPUT/.../intermediates/` without changing final TIF; diff against legacy weights in test country.
2. **Switch rasterize_kl**: read combined `W` when feature flag `agriculture.combined_proxies: true` in sector YAML.
3. **Remove dead code**: delete or archive unused `process_id` modules once α and docs are updated.
4. **Report**: `\input{chapters/04A_proxies/04A4_agriculture_combined_proxies}` from the main thesis where chapter 04A proxies are assembled.

## Open implementation details (non-methodology)

- **Bovine / dairy:** C21 column remains as configured; λ uses the same heads as today unless a separate dairy column is added later.
- **Family 5 multiband:** either one GeoTIFF per pollutant for `P_5` or a single stack with band index; must match `combine.py` and visualization.
- **Rice, urea, liming:** either extend `G` to 8+ families or keep legacy subprocess only for those pollutants until inventory mapping is decided (LaTeX fragment notes this).

## Summary

| Action | Count (approx.) |
|--------|-----------------|
| New preprocess + paths | 2–4 artefacts |
| New Python modules (`signals/`, `combine.py`) | ~6–10 files |
| Rewrite | `rasterize_kl.py`, `tabular/pipeline.py` entrypoints |
| Extend | `emission_factors.yaml`, `emission_factors.py`, `paths.yaml`, `agriculture.yaml`, visualization |
| Deprecate after switch | `enteric_grazing` / `livestock_housing` standalone scoring paths; NMVOC-only crop module; LUCAS residue in `biomass_burning` spatial leg |
