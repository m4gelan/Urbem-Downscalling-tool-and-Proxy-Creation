# CORINE morphology (GNFR C)

## Level-3 codes (defaults)

| YAML key | Default CLC L3 | Typical label (EEA) |
|----------|----------------|---------------------|
| `morphology.urban_111` | 111 | Continuous urban fabric |
| `morphology.urban_112` | 112 | Discontinuous urban fabric |
| `morphology.urban_121` | 121 | Industrial or commercial units |

Rasters are decoded with `corine.pixel_encoding` (`eea44_index` maps class indices → L3 via `PROXY/config/corine/eea44_index_to_l3.yaml`).

## Weights

Residential morphology (`residential_fireplace_heating_stove`):

`mr = w111·u111 + w112·u112 + w_other·(1 − u111 − u112 − u121)` (clipped).

Commercial morphology (`commercial_boilers`):

`mc = w111·u111 + w121·u121 + w_other·(1 − u111 − u112 − u121)` (clipped).

## Asymmetry (112 vs 121)

Residential fireplace / heating-stove bands emphasise **u112** (discontinuous urban fabric) via `w112`, reflecting stronger presence of those appliances in less-than-fully continuous urban fabric in the chosen parameterisation. Commercial boilers emphasise **u121** (industrial/commercial units) via `w121`, separating non-residential boiler stock from the same residential urban fabric signal.
