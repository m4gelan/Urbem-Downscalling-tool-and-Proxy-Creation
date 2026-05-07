# PROXY - Sector methodology reference

This document specifies, for every sector implemented in `PROXY/sectors/`, the
mathematical construction of the spatial proxy that is written to disk. Its
scope is **methodology**; the CLI, config, and outputs are documented in
`how_to_use.md`, and the refactor history is in `refactor_notes.md`.

---

## 0. Common conventions

All sectors produce a fine-grid GeoTIFF on a shared reference grid (the CORINE
raster clipped to the NUTS domain of interest). Let

- `C` = a single CAMS coarse cell,
- `pix in C` = the set of fine-grid pixels whose geographic centre falls in `C`,
- `p(pix)` = an unnormalised per-pixel score built from one or several proxies
  (OSM, CORINE, population, vessel density, LUCAS points, GFED4s, ...),
- `alpha[country, subsector, pollutant]` = a dimensionless mass share summing to
  1 over `subsector` for a given (country, pollutant).

The final per-pixel weight `w` satisfies:

```
w(pix) = p(pix) / sum_{q in C} p(q)          if sum_{q in C} p(q) > 0
w(pix) = 1 / |pix in C|                      otherwise  (uniform fallback)
sum_{pix in C} w(pix) = 1                    for every CAMS cell C
```

This is the contract enforced by `PROXY.core.raster.normalize_within_cams_cells`
and checked by `validate_weight_sums`. Pixels outside any CAMS cell carry 0 (or
the sector's declared nodata sentinel).

For multiband (per-pollutant) outputs the weight is:

```
w_pol(pix) = [ sum_s alpha[country, s, pol] * w_s(pix) ]  renormalised on C
```

where `w_s` is the normalised spatial weight for subsector `s` (derived from a
sector-specific proxy) and `alpha[country, s, pol]` is read from the
CEIP-reported-emissions workbook (see `PROXY/core/alpha/` and
`config/alpha/fallback/*.yaml`). `K_Agriculture` is the only sector whose
alpha is cross-country by design (see Section 10).

Common ingredients reused across sectors:

- `build_cam_cell_id` (`PROXY/core/cams/grid.py`): maps every fine-grid pixel
  to a CAMS-cell integer id, or `-1` if outside the CAMS grid.
- `build_p_pop` (`PROXY/core/osm_corine_proxy.py`): density-based population
  proxy = z-score of `pop / pixel_area`.
- `osm_coverage_fraction`, `rasterize_osm_coverage` (same module): burn OSM
  lines/polygons to fine-grid coverage, optionally with a buffer.
- `z_score(x)`: quantile-clipped min-max scaling (1st..99th percentile -> [0, 1]).

---

## 1. A_PublicPower

**GNFR:** A (`emission_category_index == 1`). Area + point sources; this
document covers the area output. Point matching runs through
`PROXY.core.matching`.

**Data sources**

- CAMS NetCDF (`emissions.cams_2019_nc`): bounds, country index, source-type
  index, per-source lon/lat.
- CORINE (`proxy_common.corine_tif`): filter on `area_proxy.corine_codes`
  (default `(121, 3)` in EEA44 / raw-L3 compatible form).
- JRC census population (`proxy_common.population_tif`): resampled per CAMS-cell
  clip.

**Subsectors / branches**

Single area-proxy family; the builder supports two unnormalised models selected
by `area_proxy.model`:

- `eligibility_pop_blend` (default): elig-mask + population blend.
- `corine_pop_product` (legacy): elig-mask x population^beta.

**Per-pixel weight (default model)**

For each CAMS area source `i`, clip CORINE and resample population to the
clip; define the eligibility mask `elig(pix) = 1[clc(pix) in codes]` and the
cell-local min-max of population `pop01(pix) = minmax_on_clip(max(pop, floor))`.

```
p_eligpop(pix) = a * elig(pix)^(1 + pop01(pix)) + b * pop01(pix)      (a=0.7, b=0.3 by default)
p_corinepop(pix) = elig(pix) * max(pop(pix), floor)^beta               (legacy)
```

Normalise on the clip (`share_i = p / sum p` on the cell clip). The builder
then burns each CAMS cell's shares to the reference grid with `np.add.at`. If
the CAMS cell has no eligible CORINE pixel, the logic falls back to full-cell
population, then to uniform.

**Output**: single-band float32 GeoTIFF, nodata 0. One manifest JSON alongside.

---

## 2. B_Industry

**GNFR:** B. Fully area-source.

**Data sources**

- CAMS NetCDF (CAMS-cell geometry only, via `build_cam_cell_id`).
- CORINE raster (class grid for per-group eligibility).
- Population raster (bilinear warp, density-normalised).
- OSM (`osm.industry` GPKG, *all layers concatenated* with narrowed-exception
  fallback since Phase 3.4).
- NUTS GPKG for country indexing.
- CEIP reported-emissions workbook for alpha; group definitions and OSM/CORINE
  rules in `PROXY/config/industry/ceip_groups.yaml`.

**Subsectors**

Four groups `G1..G4` (refineries, manufacturing, mineral industry,
chemical/metal); each group owns an NFR-code list (read from CEIP) and an
OSM/CORINE rule set.

**Per-pixel weight**

For each group `g`:

- `osm_raw(pix)` = subsampled coverage fraction of the matching OSM features.
- `clc_raw(pix)` = weighted score for the group's CORINE code list.
- `z_o = z_score(osm_raw)`, `z_c = z_score(clc_raw)`.
- Sector proxy: `p_sector_g = w_osm * z_o + w_clc * z_c`.
- Blend with population: `p_g(pix) = w_sector * p_sector_g + w_pop * p_pop` if
  any OSM/CORINE signal is present, else `p_g = p_pop` (fallback).

Normalise inside each CAMS cell:

```
w_g = normalize_within_cams_cells(p_g, cam_cell_id)
```

Then combine with CEIP alpha per pollutant `j` and country index `c`:

```
raw_j = sum_g alpha[c, g, j] * w_g
band_j = normalize_within_cams_cells(raw_j, cam_cell_id)         # sums to 1 per CAMS cell
```

**Output**: multiband GeoTIFF, one band per pollutant in
`industry_cfg["pollutants"]` (default 8: CH4, CO, NH3, NMVOC, NOx, PM10,
PM2_5, SOx). Sidecar CSVs: alpha tables and fallback diagnostics.

---

## 3. C_OtherCombustion

**GNFR:** C (`emission_category_index` from config, default `source_types=["area"]`).

**Data sources**

- CAMS NetCDF (per-cell emission totals and indexing).
- Hotmaps rasters (warped to the reference grid): residential heat density +
  residential gross floor area (GFA), and **separately** non-residential heat +
  non-res GFA (`heat_res` / `heat_nonres` / `gfa_res` / `gfa_nonres` in sector
  YAML under `hotmaps`).
- CORINE land-cover raster on the same grid (see **CORINE** below).
- GAINS `dom_share_ENE_*.csv` (ENE dominant-share by region).
- JSON: `EMEP_emission_factors.json`, `GAINS_mapping.json`,
  `config/sectors/other_combustion/eurostat_end_use.json` (sidecar for Eurostat
  parsing).
- Eurostat dissemination API (`nrg_d_hhq`, `nrg_bal_s`) when `eurostat.enabled` is true,
  with JSON cache under `PROXY/cache/eurostat/` (no XLSX scraping).

**Subsectors (appliance axis)**

Seven activity archetypes `K = 7`:

```
R_FIREPLACE, R_HEATING_STOVE, R_COOKING_STOVE, R_BOILER_MAN, R_BOILER_AUT,
C_BOILER_MAN, C_BOILER_AUT
```

The leading `R_` / `C_` prefix is **residential vs commercial** in the *model*
sense. GAINS rows are mapped onto class `k` via `GAINS_mapping.json`; for each
matching row with normalized activity share `share`, pollutant `p`, and EMEP
EF `EF(p, fuel, appliance)` in kg/TJ:

```
M[p, k] += share * f_enduse[bucket(k)] * f_appliance[k] * EF(p, fuel, appliance)
```

(summed over all GAINS rows that map to the same class). **End-use multipliers**
`f_enduse` are documented next; they scale activity **before** EF application in
this product form.

### End-use factors `f_enduse` (Eurostat)

`f_enduse` is a per–model-class scalar used inside `build_M_for_country` when
turning GAINS activity into pollutant-specific terms. It does **not** split
pixels between residential and commercial; that split comes from Hotmaps +
CORINE in the spatial stack `X`.

**Defaults**

- If `eurostat.enabled` is false, **end-use bucket weights** and **appliance splits**
  are all **1.0** (same net effect as the former per-class `f_enduse = 1.0`).

**When Eurostat is enabled**

1. **Sidecar** (`eurostat_end_use.json`): supplies `year`, `iso3_to_geo_labels`
   (Eurostat **geo** code, typically 2 letters, e.g. `EL` for Greece), and
   `class_to_metric` mapping each `R_*` class to one or more **metric keys**
   (`space_heating`, `water_heating`, `cooking`, …). `metric_row_synonyms` is
   retained for documentation only (API uses fixed `nrg_bal` codes).
2. **API `nrg_bal_s`**: `FC_OTH_CP_E / (FC_OTH_CP_E + FC_OTH_HH_E)` defines the
   **commercial energy share** `α`. Household end-use TJ from **`nrg_d_hhq`**
   (`FC_OTH_HH_E_SH`, …, `FC_OTH_HH_E_OE`) are turned into **shares within
   households**, then multiplied by **`(1 − α)`** so residential buckets sum to
   the household slice of the national budget.
3. **GAINS appliance split** `f_appliance[class]`: for each end-use **bucket**
   implied by `class_to_metric`, GAINS row shares are summed per `MODEL_CLASSES`
   column and **normalised within the bucket** so appliances do not double-count
   the same Eurostat mass.
4. **Matrix term**: each GAINS row still contributes  
   `share × f_enduse[bucket] × f_appliance[class] × EF`  
   with **`C_*`** classes using the **`commercial`** bucket.

Legacy XLSX parsing (`table_3_country_rows`, etc.) was **removed**; the sidecar
keys remain only for geo/year/class mapping.

### CORINE

1. The CORINE band configured under `corine.band` is warped to the reference
   grid with **nearest** neighbour resampling (class values must stay discrete).
2. **Pixel encoding** (`corine.pixel_encoding`):
   - `l3_code`: raster values are already CORINE Level-3 codes.
   - `eea44_index` (common for EEA-style products): values are 1-based class
     indices; they are converted to L3 using `PROXY/config/corine/eea44_index_to_l3.yaml`
     (or `corine.pixel_value_map` if overridden).
3. For the three Level-3 codes in `morphology` (`urban_111`, `urban_112`,
   `urban_121`, default **111 / 112 / 121**), `build_clc_indicators` builds binary
   masks `u111`, `u112`, `u121` (1 where that L3 class, else 0). The remainder
   `1 - u111 - u112 - u121` (clipped to [0, 1]) is “other” urban/context within
   the pixel for morphology weighting only.

CORINE does **not** enter `M` directly; it modulates **where** residential vs
commercial *spatial* mass goes (next subsection).

### Residential vs commercial **spatial** signal (Hotmaps + CORINE)

**Two base intensity fields** (`preprocess.combine_base` on warped Hotmaps):

- **`R_base`**: combines **residential** heat density and **residential** GFA
  (either multiplicative exponents or a weighted additive blend from
  `base_proxy` in YAML).
- **`C_base`**: same construction from **non-residential** heat and **non-res**
  GFA.

So the **primary** residential / commercial distinction on the map is
**Hotmaps’ res vs non-res products**, not CORINE land-cover class alone.

**CORINE morphology multipliers** (`morph_residential` / `morph_commercial`):

- **Residential** (`morphology.residential_fireplace_heating_stove`):  
  `mr = w111*u111 + w112*u112 + w_other*(1 - u111 - u112 - u121)` (clipped).
- **Commercial** (`morphology.commercial_boilers`):  
  `mc = w111*u111 + w121*u121 + w_other*(1 - u111 - u112 - u121)` (clipped).  
  Note the asymmetry: residential fireplace/heating stove stresses **u112**
  (discontinuous urban fabric) with weight `w112`; commercial boilers stress
  **u121** (industrial / commercial units) with `w121`, and do not use `u112` in
  the same slot.

**Stack `X`** (`build_X_stack`, shape H × W × 7):

| Band (class)        | Formula        | Interpretation |
|---------------------|----------------|----------------|
| `R_FIREPLACE`, `R_HEATING_STOVE` | `R_base * mr` | Residential density × residential morphology |
| `R_COOKING_STOVE`, `R_BOILER_MAN`, `R_BOILER_AUT` | `R_base` | Residential density only (no `mr`) |
| `C_BOILER_MAN`, `C_BOILER_AUT` | `C_base * mc` | Non-res density × commercial morphology |

GAINS + EMEP + `f_enduse` define **`M[pollutant, k]`**; **`X[pix, k]`** is the
spatial prior above. For CAMS cell `i`, overlapping ref pixels form `X_w`;
**`U = X_w @ M.T`**; each pollutant column is normalized to shares over pixels
(uniform fallback if the sum is non-positive or non-finite), then multiplied by
that cell’s CAMS emission and accumulated (`np.add.at`). Weight GeoTIFFs stack
those shares across pollutants.

**Output**: multiband GeoTIFF, `count = len(pollutant_specs)`; band names
`weight_share_gnfr_c_<pollutant>`. Optional single-band per-pollutant TIFs and
absolute-emission TIFs if the sector YAML asks for them.

---

## 4. D_Fugitive

**GNFR:** D. Shares the structural layout of `B_Industry` almost verbatim.

**Data sources**

- CAMS NetCDF (cell geometry / CAMS-id).
- CORINE raster.
- Population raster.
- OSM (`osm.fugitive` GPKG) filtered by `industrial`, `landuse`, `man_made`,
  `resource`, `power`, `amenity`.
- NUTS GPKG for `rasterize_country_ids`.
- CEIP workbook for alpha; group definitions in
  `PROXY/config/fugitive/ceip_groups.yaml`; optional YAML fallback
  `PROXY/config/alpha/fallback/D_Fugitive_*.yaml`.

**Subsectors**

Four groups `G1..G4` (coal / mines / quarries, oil / pipelines / ports,
storage / refining, gas / flaring ...) mapped to NFR CEIP codes
(`1B1A`, `1B2AI`, `1B2AIV`, `1B2B`, ...).

**Per-pixel weight**

Identical structure to `B_Industry`:

```
p_g = blend( w_osm * z(osm_g) + w_clc * z(clc_g),  p_pop )        default w_sector=0.8, w_pop=0.2
w_g = normalize_within_cams_cells(p_g, cam_cell_id)
lambda_j(pix) = sum_g alpha[c, g, j] * w_g(pix)
band_j = normalize_within_cams_cells(lambda_j, cam_cell_id)       # sums to 1 per CAMS cell
```

**Output**: multiband GeoTIFF, one band per pollutant (same default list as
`B_Industry`), plus CSV sidecars for alpha, fallback counts, and per-cell pop
fallbacks.

---

## 5. E_Solvents

**GNFR:** E (`IDX_E = 5`, area sources `IDX_AREA = 1`).

**Data sources**

- CAMS NetCDF (`cell_of` mapping for GNFR-E area sources).
- CORINE raster (residential, urban-fabric, service, industrial CLC lists).
- Population raster (sum-resampled).
- OSM (`osm.solvents` GPKG): `osm_landuse`, `osm_buildings`, `osm_aeroway`,
  `osm_roads` (with PBF fallbacks).
- CEIP workbook + `PROXY/config/ceip/profiles/solvents_subsectors.yaml`.

**Subsectors**

Nine 2.D.3 activity keys: `d3a_domestic ... d3i_other` (see `defaults.json`),
mapped from NFR tokens such as `2D3A`, `2D3B`, ...

**Per-pixel weight**

Four archetype proxies from quantile-clipped indicators:

```
rho_house, rho_serv, rho_ind, rho_infra in [0, 1]
```

Each subsector `s` mixes archetypes:

```
rho_raw_s(pix) = sum_k  beta[s, k] * max(rho_arche_k, 0)
```

Within a CAMS parent `i`:

```
rho_norm_s(pix) = rho_raw_s / sum_{q in i} rho_raw_s        (primary)
            or   rho_gen / sum_{q in i} rho_gen             (if primary denom = 0)
            or   uniform                                    (if generic also 0)
```

Finally combine with CEIP mass-share alpha:

```
W_p(pix) = sum_s  alpha[p, s] * rho_norm_s(pix)
```

A `PROXY.core.raster.validate_parent_weight_sums_strict` pass confirms
`sum_{pix in i} W_p(pix) = 1`.

**Output**: multiband GeoTIFF, `P = len(pollutants)` = 6 by default (CO, NOx,
NH3, NMVOC, PM10, PM2_5); band names `W_<pollutant>_GNFR_E_area`. Optional
JSON sidecar with per-cell mass-balance diagnostics.

---

## 6. G_Shipping

**GNFR:** G (`gnfr_g_index`, default 10). Single shared proxy for every
pollutant (no CEIP per-pollutant alpha at the spatial stage).

**Data sources**

- CAMS NetCDF (`build_cam_cell_id` only).
- EMODnet vessel density GeoTIFF
  (`proxy_specific.shipping.vessel_density_tif`), bilinear-warped.
- CORINE raster: nearest class grid + average-resampled port mask (default
  CLC level-2 `123`); water CLC set `35..44` for land damping.
- OSM (`osm.shipping` GPKG): layers `osm_shipping_high`,
  `osm_shipping_medium` (medium drops `industrial_shipyard` and
  `landuse_industrial`).

**Subsectors**

None at the spatial stage. A single shared `G_Shipping` proxy.

**Per-pixel weight**

```
D_n    = vessel_density / max(vessel_density)
damp   = 1 on water or port CLC, else land_damp (default 0.12)
D_land = D_n * damp
z_osm  = minmax01(osm_coverage)
z_port = minmax01(port_fraction)
P_raw  = a0 * D_land + a1 * z_osm + a2 * z_port             (a0+a1+a2 = 1)
```

CAMS-cell normalisation:

```
w(pix) = P_raw(pix) / sum_{q in C} P_raw(q)                 (uniform fallback if sum = 0)
```

Pixels with `cam_cell_id < 0` carry the nodata sentinel `-9999`.

**Output**: single-band float32 GeoTIFF, band name `g_shipping_weight`,
nodata `-9999`. Optional diagnostic rasters (`emodnet_raw`,
`D_n_damped`, `osm_coverage`, `port_fraction`).

---

## 7. I_Offroad

**GNFR:** I (NFR codes `1A3c` rail, `1A3ei` pipeline, `1A3eii` non-road).

**Data sources**

- CAMS NetCDF (`build_cam_cell_id`).
- CORINE raster (`PROXY.core.corine_masks.warp_corine_codes_nearest`).
- NUTS GPKG (`rasterize_country_indices` for per-pixel ISO3 / CEIP rows).
- OSM (`osm.offroad` GPKG):
  - `osm_offroad_rail_lines` filtered by `osm_railway_line_filter_sets`.
  - `osm_offroad_areas` with `offroad_family == "landuse_railway"`.
  - `osm_offroad_pipeline_lines` and `offroad_family == "man_made_pipeline"`
    areas.
- CEIP workbook (`read_ceip_shares`) for the triple share per country x pollutant.
- Optional `area_proxy.facilities_tif` for pipeline z-blend.
- YAML override: `PROXY/config/alpha/fallback/I_Offroad_*.yaml`.

**Subsectors**

Three legs:

- `rail`     (1A3c)
- `pipeline` (1A3ei)
- `nonroad`  (1A3eii)

For each country and pollutant, CEIP yields
`(s_rail, s_pipe, s_nonroad)` with `s_rail + s_pipe + s_nonroad = 1`.
YAML alpha fallbacks are applied only where CEIP is missing or equal to the configured
default triple.

**Per-pixel weight**

```
z_rail    = z_score( cov_rail_lines + cov_landuse_railway )
z_pipe    = z_score( cov_pipeline_lines + cov_pipeline_areas )
z_pipe'   = w_raw * z_pipe + w_fac * z_score(facilities)           (optional)
p_nonroad = w_agri * z_score(corine_agri) + w_ind * z_score(corine_industrial)
p(pix)    = s_rail(c,p) * z_rail + s_pipe(c,p) * z_pipe' + s_nr(c,p) * p_nonroad
w(pix)    = normalize_within_cams_cells(p, cam_cell_id)            # uniform fallback per cell
```

**Output**: multiband GeoTIFF, one band per pollutant (default `nox`, `pm2_5`,
`nh3`, `co`, `nm_voc`, `so2`); band names = normalised pollutant keys.

---

## 8. J_Waste

**GNFR:** J (`gnfr_j_index`, default 13). Produces two GeoTIFFs:
`waste_areasource.tif` (source-type 1) and `waste_pointsource.tif`
(source-type 2).

**Data sources**

- CAMS NetCDF (area / point CAMS cell masks, country index).
- CEIP workbook for family mass shares (`PROXY/config/waste/ceip_families.yaml`).
- CORINE, Population, NUTS.
- OSM (`osm.waste` GPKG): `osm_waste_polygons` / `osm_waste_points` with
  `waste_family`.
- `proxy_specific.waste`: `impervious_tif` (required), `ghsl_smod_tif`,
  `agglomerations_gpkg` (UWW), `treatment_plants_gpkg` (WWTP).
- Optional: `ref_tif`, `point_source_mask_tif`.

**Subsectors**

Three families assigned by CEIP row token:

- `solid` (`5A`, `5B1`, `5B2`)
- `ww`    (`5D1`, `5D2`, `5D3`)
- `res`   (`5C2`, `5E`)

`5C1*` (incineration) is in `exclude_always`.

**Per-pixel weight**

Build one raw proxy per family:

- `P_solid` = weighted CORINE 121/132 (via `adapt_corine_classes_for_grid`) +
  optional OSM landfill / amenity layers; normalised to [0, 1].
- `P_ww`    = UWW agglomeration union + WWTP buffers (+ optional Gaussian) +
  population + impervious + industrial CORINE 131/132/133; weighted sum then
  normalised.
- `P_res`   = population + rural GHSL-SMOD mask + impervious settlement;
  weighted sum then normalised.

Compose per pollutant `p` using CEIP mass shares
`w_s, w_w, w_r` and masks `m_s = 1[P_solid > eps]` etc.:

```
a = w_s * m_s ,  b = w_w * m_w ,  c = w_r * m_r
P(pix, p) = (a * P_solid + b * P_ww + c * P_res) / (a + b + c)        if denom > 0
          = mean of active families                                    if any mask active
          = 0                                                         otherwise
```

Apply an optional point-source mask (for `waste_areasource.tif` this clears
the point cells, and conversely for the point stream). Then normalise
per CAMS cell separately on the area- and point-cell rasters:

```
w(pix, p) = P / sum_{q in C} P                 (uniform fallback if sum = 0)
```

**Output**: two multiband GeoTIFFs (area and point), `count = len(pollutants)`
= 8 by default. Band names `j_waste_weight_<pollutant>`.

---

## 9. K_Agriculture (U120 = forest, user-confirmed 2026-04)

**GNFR:** K and L (categories 14 and 15 in CAMS, `source_type == 1`).

**Data sources**

- LUCAS 2022 CSV (`EU_LUCAS_2022.csv`): `SURVEY_LC1`, `SURVEY_LU1`,
  `SURVEY_GRAZING`, residue / NMVOC fields, `SURVEY_LC1_PERC`.
- LUCAS Soil 2018 CSV (`pH_H2O`, `OC`, `LC`) for liming only.
- CORINE raster: class sampling at LUCAS points + per-NUTS2 x CLC pixel counts
  for the agricultural band (EEA44 indices 12..22 = CLC L3 codes 211..244).
- NUTS GPKG (`NUTS_ID`, `CNTR_CODE`, Level-2 polygons).
- C21 GPKG (national census of livestock head counts), schema in
  `c21_census_field_map.json`.
- GFED4.1s preprocessed arrays for biomass burning.
- JSON-backed emission factors under
  `PROXY/config/agriculture/emission_factors/*.json` (Phase 1.5).
- YAML: `PROXY/config/agriculture/lu1_lc1_mapping.yaml` (LC1/LU1/GRAZING
  interpretation, including the **disabled-by-default** U120 opt-in switch).
- CAMS NetCDF via `PROXY.core.cams.build_cams_source_index_grid_any_gnfr`
  (GNFR K+L area-source filter).

**Subprocesses -> pollutants**

| `process_id`           | Pollutants                              |
|-------------------------|-----------------------------------------|
| `enteric_grazing`       | CH4                                     |
| `manure`                | CH4, NH3, NOx, NMVOC                    |
| `fertilized_land`       | NH3, NOx                                |
| `rice`                  | CH4                                     |
| `biomass_burning`       | NOx, NMVOC, CO                          |
| `livestock_housing`     | PM10, PM2_5                             |
| `agricultural_soils`    | NMVOC (crop NMVOC / 3.D)                |
| `soil_liming`           | CO2                                     |
| `urea_application`      | CO2                                     |

Alpha shares are **cross-country constants** (no CEIP workbook), read from
`PROXY/config/agriculture/alpha.config.json`.

**Per-pixel weight chain**

For a given subprocess, let `s_p` be the per-LUCAS-point score.

Stage 1, point-level score: per-subprocess rule on LUCAS columns. Examples:

```
enteric_grazing:      s_p = 1 if SURVEY_GRAZING == 1 else (0.6 if GRAZING missing and LC1 in {E*, B55} else 0)
manure_land:          s_p = max( grazing_metric, land_app_weight(LU1, LC1) )
livestock_housing:    s_p = tier(LU1, LC1, GRAZING)       # None for LU1 = U120 (forest)
fertilized_land:      s_p = synthetic_N_rate_for_lc1(LC1)  if eligible(LC1, LU1) else NaN
urea_application:     s_p = 1 on cropland, 0.7 * omega_grass(NUTS) on grassland
rice:                 s_p = 1 if LC1 == B17, else excluded
biomass_burning:      custom (see below)
agricultural_soils:   s_p = EF_NMVOC(LC1) * LC1_PERC / 100
soil_liming:          s_p = lime_score(pH_H2O, OC, LC)
```

Stage 2, (NUTS-2, CLC) aggregation:

```
mu[n, c] = mean_{p in group(n, c)} s_p
```

Optional stage-3 census multiplier (enteric and housing):

```
mu <- mu * omega(n)                                       # omega from C21 / NUTS head-counts
```

Country-level normalisation yields `rho`:

```
rho[n, c] = mu[n, c] / max_{n', c' in same country} mu[n', c']
```

Biomass burning uses a special `mu`:

```
mu_burn(n)      = sum_{pix in n}  DM_mean(pix) * area(pix)
w_bar[n, c]     = mean_{p in LUCAS(n, c)} ( residue_ratio(LC1) * LC1_PERC )
weighted_pix    = n_pixels[n, c] * w_bar[n, c]
mu[n, c]        = mu_burn(n) * weighted_pix / sum_c' weighted_pix
rho[n, c]       = mu[n, c] / max_country(mu)
```

Tabular pollutant score and NUTS-level weights:

```
S[n, c, pol] = n_pixels[n, c] * sum_s  alpha[pol, s] * rho[n, c, s]
w[n, c, pol] = S[n, c, pol] / sum_c  S[n, c, pol]               # per NUTS_ID, per pollutant
```

Raster propagation and CAMS-source normalisation (`rasterize_kl.build_kl_sourcearea_tif`):

```
raw(pix, pol) = w[NUTS(pix), CLC(pix), pol]                     # agricultural CLC only
den(C, pol)   = sum_{q in C} raw(q, pol)
out(pix, pol) = raw(pix, pol) / den(C, pol)                     if den > 1e-30
out(pix, pol) = 1 / n_ag_pixels_in_C                            if den = 0 and at least one ag pixel
out(pix, pol) = 0                                               otherwise
```

The per-CAMS-cell sum therefore equals 1 for every cell that touches the
agricultural domain, consistent with the general contract (Section 0).

**Output**: multiband float32 GeoTIFF, one band per pollutant in
`sector_cfg["raster_pollutants"]` (default: CH4, NH3, NOx); band names
`weight_share_agri_<POLLUTANT>`. Sidecar JSON manifest.

The active build entrypoint is `K_Agriculture.builder -> K_Agriculture.pipeline`;
the tabular model lives under `K_Agriculture.tabular`, while process-specific
methodology stays in `K_Agriculture.source_relevance`.

**Note on U120**: per user confirmation (2026-04) U120 is treated as forest
and excluded from every agricultural proxy. The research-mode YAML switch
`livestock_housing.u120_mixed_livestock.enabled` ships `false`. See
`docs/agriculture_class_mapping.md` section 1.1.

---

## 10. Why `K_Agriculture`'s alpha is cross-country

`K_Agriculture` does not consume the CEIP reported-emissions workbook at
all; its alpha shares are fixed constants in `alpha.config.json`. The reason is
methodological: the CAMS-REG-ANT agriculture methodology ties `alpha` to
process-level emission factors (enteric / manure / soils / ...) that are not
per-country in the underlying IPCC defaults. Every other sector pulls `alpha`
from CEIP, with YAML-based cross-country defaults in
`config/alpha/fallback/defaults.yaml` and per-country overrides under
`config/alpha/fallback/<SECTOR>_<ISO2>.yaml`.

---

## 11. Summary table

| Sector          | GNFR  | Per-sector alpha? | Pollutants (default bands) | Output shape                         |
|-----------------|-------|-------------------|----------------------------|--------------------------------------|
| A_PublicPower   | A     | n/a (area proxy)  | single (proxy)             | 1 band                               |
| B_Industry      | B     | CEIP, 4 groups    | 8                          | multiband per pollutant              |
| C_OtherCombustion | C   | CAMS emission x EF | variable (pollutant_specs) | multiband per pollutant              |
| D_Fugitive      | D     | CEIP, 4 groups    | 8                          | multiband per pollutant              |
| E_Solvents      | E     | CEIP, 9 keys      | 6                          | multiband per pollutant              |
| G_Shipping      | G     | n/a (shared proxy) | single                    | 1 band                               |
| I_Offroad       | I     | CEIP, 3 legs      | 6 (default)                | multiband per pollutant              |
| J_Waste         | J     | CEIP, 3 families  | 8                          | 2 multiband files (area + point)     |
| K_Agriculture   | K+L   | cross-country     | 3 (default)                | multiband per pollutant              |

Every multiband file follows Section 0's contract: `sum over pixels in a CAMS
cell = 1` for every band.
