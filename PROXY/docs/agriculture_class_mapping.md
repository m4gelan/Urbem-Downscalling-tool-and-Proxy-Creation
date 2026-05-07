# Agriculture class mapping (LUCAS LC1 / LU1 / GRAZING + CORINE)

This document is the authoritative reference for every agricultural class code that the
`K_Agriculture` pipeline interprets. It is paired with a machine-readable version at
`PROXY/config/agriculture/lu1_lc1_mapping.yaml`. Whenever the classification of a code
changes, **both files must be updated in lockstep** -- the YAML is the single source of
truth for runtime decisions, this document is the single source of truth for methodology.

Sources:
- LUCAS 2022 Technical Reference Document C3 (land cover + land use dictionaries).
- EMEP/EEA air pollutant emission inventory guidebook 2023 (EFs for biomass burning,
  NMVOC crop production, manure management -- Velthof et al. 2009).
- Thunen NIR (assignment of NMVOC EFs to LUCAS LC1 codes).
- Goulding (2016) / AHDB RB209 (lime demand by pH).

## 1. LUCAS SURVEY_LU1 codes used in PROXY

| Code  | LUCAS meaning                                             | PROXY interpretation                                                       | Used by                                 |
|-------|-----------------------------------------------------------|----------------------------------------------------------------------------|------------------------------------------|
| U111  | Agriculture (excluding fallow land)                       | Agricultural land, active cultivation / management                         | enteric, manure, housing, soils, urea, fertilized_land |
| U112  | Fallow land                                               | Agricultural land, fallow in the survey year                               | enteric, manure, housing, soils, urea, fertilized_land |
| U113  | Kitchen gardens                                           | Small-scale / non-commercial agriculture                                   | manure (downweighted)                    |
| U120  | **Forest (user-confirmed 2026-04)**                       | Excluded from every agricultural proxy (housing/urea/manure/fertilizer). Matches the legacy PROXY behaviour. A YAML opt-in (`livestock_housing.u120_mixed_livestock.enabled`) is available for research-mode "mixed farming" scenarios but ships disabled - see Section 1.1. |
| U130  | Hunting and fishing                                       | Not agricultural for PROXY purposes                                        | (not mapped)                             |

### 1.1 U120 = forest (user-confirmed, production default)

The user confirmed on 2026-04 that `U120` represents forest land in this pipeline
and must remain excluded from every agricultural proxy. The production default
reproduces the legacy `livestock_housing.py` behaviour exactly: any LUCAS point with
`SURVEY_LU1 == "U120"` returns `None` from the point-level scorer and therefore
contributes nothing to housing / manure / fertilized-land / urea rho.

Phase 1.3 of the refactor also introduced a YAML switch
(`livestock_housing.u120_mixed_livestock` in `lu1_lc1_mapping.yaml`). This is an
opt-in knob for research scenarios where a country-specific reinterpretation of
`U120` as "mixed farming with livestock dominant" is desired. It is **disabled**
by default (`enabled: false`, `score: null`) and must stay disabled for production
runs. If a user enables it, `livestock_housing.py` emits a WARNING-level log line
so that the non-default scoring is never silent, and the configured `score` is
applied to U120 points only (all other classes are untouched).

## 1.2 Phase 1.4 - Fertilized-land LU1 audit (documented, no numeric change)

`source_relevance/fertilized_land.py::eligible_synthetic_n` (L64-75) returns `True`
whenever LC1 starts with `B`, without inspecting LU1. This over-includes kitchen gardens
(`U113`), fallow land (`U112`), and any LUCAS point with LU1=`U120` whose cover happens to
start with `B` (even though U120 is forest). The E20 grassland branch, by contrast,
already requires `LU1 == U111` -- the handling is asymmetric.

The YAML (`fertilized_land.phase_1_4_audit`) lists candidate restrictions:

1. Restrict B* to `LU1 in {U111, U112}`.
2. Restrict B* to `LU1 == U111` only (matches the E20 pattern).
3. Keep current behaviour as intentional over-inclusion.

No numeric change is applied until the user picks one of these; all three would shift
synthetic-N mu and therefore NH3 / NOx / N2O rho.

## 2. LUCAS SURVEY_LC1 codes used in PROXY

### 2.1 B-series (cropland)

| Code  | LUCAS meaning                                     | PROXY use                                       |
|-------|---------------------------------------------------|-------------------------------------------------|
| B11   | Common wheat                                      | manure high group (0.8); NMVOC 0.32; residue 1.3 |
| B12   | Durum wheat                                       | manure intermediate (0.6); NMVOC 0.32; residue 1.3 |
| B13   | Barley                                            | manure high (0.8); NMVOC 1.03; residue 1.2      |
| B14   | Rye                                               | manure intermediate (0.6); NMVOC 0.32; residue 1.6 |
| B15   | Oats                                              | manure intermediate (0.6); NMVOC 0.32; residue 1.3 |
| B16   | Maize                                             | manure intermediate (0.6); NMVOC 0.32; residue 1.0 |
| B17   | Rice                                              | manure intermediate (0.6); NMVOC 0.32; residue 1.4; **rice paddy proxy = 1.0** |
| B18   | Triticale                                         | manure intermediate (0.6); NMVOC 0.32; residue 1.2 |
| B19   | Other cereals                                     | manure intermediate (0.6); NMVOC 0.32; residue 1.2 |
| B21   | Potatoes                                          | manure residual-B (0.1); NMVOC 0.32             |
| B22   | Sugar beet                                        | manure residual-B (0.1); NMVOC 0.32             |
| B23   | Other root crops                                  | manure residual-B (0.1); NMVOC 0.32             |
| B2*   | All "oil crops" in Velthof (2009)                 | manure high (0.8, via `startswith("B2")`)       |
| B31   | Sunflower                                         | manure intermediate (0.6); NMVOC not listed     |
| B32   | Rape / turnip rape                                | manure high (0.8); NMVOC 1.34                   |
| B33   | Soya                                              | residue 2.1                                     |
| B41   | Dry pulses                                        | NMVOC 0.32; residue 1.7                         |
| B51   | Clover                                            | NMVOC 0.41                                      |
| B52   | Lucerne                                           | NMVOC 0.41                                      |
| B53   | Other legumes                                     | NMVOC 0.41                                      |
| B54   | Mixed legumes + grasses                           | NMVOC 0.41                                      |
| B55   | **Ley (temporary grassland)**                     | NMVOC 0.41; **housing tier 0.30** (ley-specific); **manure** uses the same `startswith("B")` path as arable (flagged for Phase 1.4 audit); **grazing** returns 0.6 when SURVEY_GRAZING is missing |

### 2.2 E-series (grassland and woodland)

| Code | LUCAS meaning         | PROXY use                                                                      |
|------|-----------------------|--------------------------------------------------------------------------------|
| E10  | Grassland with trees  | NMVOC 0.41; fertilized_land eligible; manure 0.8 (managed grassland); urea 0.7*omega |
| E20  | Grassland without trees | NMVOC 0.41; fertilized_land eligible **only if LU1 == U111**; otherwise excluded |

### 2.3 A-series (artificial / buildings)

| Code | LUCAS meaning  | PROXY use                                                         |
|------|----------------|-------------------------------------------------------------------|
| A11  | Buildings - roof | livestock_housing tier 1.0 when LU1 == U111                      |
| A12  | Buildings - wall | livestock_housing tier 1.0 when LU1 == U111                      |

Other A-series codes are treated as non-agricultural (no housing / manure score).

## 3. LUCAS SURVEY_GRAZING codes

| Code | LUCAS meaning                       | PROXY use                                      |
|------|-------------------------------------|-------------------------------------------------|
| 0    | No grazing evidence                 | housing confirmed-non-grazing (full tier); manure component = 0; enteric = 0 |
| 1    | Grazing observed                    | housing = 0 (incompatible with confinement); manure = 1.0; enteric = 1.0 |
| 2    | Grazing traces but no animals seen  | housing confirmed-non-grazing; manure component = 0; enteric = 0 |
| NaN  | Missing                             | housing damped (0.4 x tier); manure and enteric fall back to LC1-based heuristic (0.6 when LC1 startswith "E" or == "B55") |

## 4. CORINE CLC categories used by the agricultural pipeline

`K_Agriculture` operates on CORINE EEA44 indices 12-22 (the agricultural CLC classes):

| EEA44 index | CLC L3 code | CLC L3 label                                  |
|-------------|-------------|------------------------------------------------|
| 12          | 211         | Non-irrigated arable land                     |
| 13          | 212         | Permanently irrigated land                    |
| 14          | 213         | Rice fields                                   |
| 15          | 221         | Vineyards                                     |
| 16          | 222         | Fruit trees and berry plantations             |
| 17          | 223         | Olive groves                                  |
| 18          | 231         | Pastures                                      |
| 19          | 241         | Annual crops associated with permanent crops  |
| 20          | 242         | Complex cultivation patterns                  |
| 21          | 243         | Land principally occupied by agriculture, with significant areas of natural vegetation |
| 22          | 244         | Agro-forestry areas                           |

## 5. Process / alpha mapping

The `PROXY/config/agriculture/alpha.config.json` file assigns a fixed alpha share to each
`process_id`. Every `process_id` is implemented by one module in
`PROXY/sectors/K_Agriculture/source_relevance/` (see `alpha.config.json` `module` field).

Alpha values are cross-country constants by design: the CAMS-REG-ANT methodology for
agriculture does not use reported-emissions national totals (unlike Industry / Fugitive /
Waste / Solvents). See `PROXY/config/alpha/fallback/defaults.yaml` (section `K_Agriculture`
is intentionally empty) and `docs/refactor_notes.md` for the reasoning.
