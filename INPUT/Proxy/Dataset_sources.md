# Dataset Sources for GNFR Proxy Construction

This file is a reference for the datasets used to build the spatial proxies associated with the different GNFR sectors. It is meant to answer four questions for each dataset:

- what the dataset contains;
- why it is used in the proxy workflow;
- where it can be downloaded;
- whether access is open or requires registration.

The descriptions below are intentionally practical: they focus on how the datasets are used in this repository rather than on full metadata documentation.

## Cross-sector datasets

These datasets are reused across several sectors.

### Boundaries

**Dataset:** GISCO NUTS 2021 boundaries (`NUTS_RG_03M_2021_3035.gpkg`)

**Description:** European administrative boundaries distributed by GISCO, including countries and NUTS regions in `EPSG:3035`. In this project, the layer is primarily used to derive country, NUTS2, and other regional masks for clipping, aggregation, and reporting.

**Access:**
- Direct file: https://gisco-services.ec.europa.eu/distribution/v2/nuts/gpkg/NUTS_RG_03M_2021_3035.gpkg
- Overview page: https://gisco-services.ec.europa.eu/distribution/v2/nuts/nuts-2021-files.html

**Access conditions:** no account required.

### CORINE Land Cover

**Dataset:** CORINE Land Cover 2018 (vector and raster, 100 m)

**Description:** Harmonised European land cover classification produced under Copernicus Land Monitoring Service. It provides a consistent land cover map for Europe and is used here to identify broad land classes, especially when a sector proxy depends on distinguishing agricultural, urban, forest, industrial, or other land-use types.

**Main characteristics:**
- European coverage
- 100 m raster product, with matching vector product
- 6-year update cycle
- distributed in `EPSG:3035`

**Access:** https://land.copernicus.eu/en/products/corine-land-cover/clc2018#download

**Access conditions:** Copernicus account may be required for download.

### E-PRTR / Industrial reporting

**Dataset:** European Pollutant Release and Transfer Register / Industrial Reporting under the Industrial Emissions Directive

**Description:** European inventory of major industrial facilities and their reported releases, transfers, and related industrial reporting attributes. In this repository it is mainly used as a point-based source of industrial facility locations and characteristics for industry-related proxy construction.

**Recommended dataset entry:**
- `Industrial Emissions Directive 2010/75/EU and European Pollutant Release and Transfer Register Regulation (EC) No 166/2006 - ver. 15.0 Dec. 2025 (Tabular data)`

**Access:** https://www.eea.europa.eu/en/datahub/datahubitem-view/9405f714-8015-4b5b-a63c-280b82861b3d

**Access conditions:** no account required.

### OpenStreetMap

**Dataset:** OpenStreetMap (OSM)

**Description:** Collaborative global geospatial dataset containing roads, railways, land-use polygons, buildings, waterways, ports, and many other mapped features. In this project, OSM is used to derive sector-specific feature layers from a continental extract, especially when the proxy relies on linear infrastructure or detailed land-use polygons not available in a single European thematic product.

**Project-specific workflow:**
- the European extract is downloaded with the link below;
- sector-specific layers are then generated with scripts of the form under `Prepocess/OSM`;
- the outputs are stored as `{sector}_layers.gpkg`.

**Access:** https://download.geofabrik.de/europe.html

**Access conditions:** no account required for using the data; extracts may come from external providers depending on the download workflow.

### Population

**Dataset:** JRC-ESTAT Census Population Grid 2021

**Description:** High-resolution gridded residential population counts for Europe at 100 x 100 m resolution. The product is derived from the 2021 census grid through dasymetric disaggregation and is useful wherever population distribution is needed as a spatial proxy, especially for residential or service-related activities.

**Main characteristics:**
- 100 m raster grid
- residential population counts
- European coverage
- suitable for high-resolution overlay with land-use and infrastructure data

**Access:** https://data.jrc.ec.europa.eu/dataset/98336641-fd1c-4992-8c7b-c470dd5eb81e

**Access conditions:** no account required.

## Proxy-specific datasets

### Agriculture

#### VIIRS I-band 375 m active fire (FIRMS)

**Dataset:** NASA VIIRS thermal anomalies / active-fire product distributed for fire-management and science applications (375 m nominal at nadir, Suomi NPP / NOAA-20 / NOAA-21 heritage). Operational GIS-friendly archives are disseminated via **FIRMS** (Fire Information for Resource Management System); see the NASA Earthdata instrument overview for context and tooling links.

**Description:** Near-daily hotspot detections with location, brightness temperatures, acquisition time, satellite, confidence flag, daytime/night indicator, fire radiative power (FRP), and related QA fields. Here it replaces legacy coarse-resolution burned-area composites for `K_Agriculture` **open biomass burning (3.F)**: VIIRS points are buffered and intersected with a CORINE-derived cropland mask so seasonal burning activity aligns with agricultural land rather than unrelated wildfires alone.

**Main characteristics:**
- global coverage from polar-orbit VIIRS (~375 m I-band detections at nadir; footprint grows off-nadir)
- sub-daily revisit at mid-latitudes (multiple satellites)
- attribute set documented by NASA/FIRMS (e.g. Latitude/Longitude, Acq_Date/Time, FRP, Confidence, Satellite, Bright_ti4/ti5, Scan/Track, DayNight); pipeline reads a subset exported to shapefile — see below

**Project retrieval (bulk / area extract):**
- **Geographic extent:** Europe (bounding box supplied with the download request).
- **Start date:** 2015-01-01
- **End date:** 2022-01-01
- Delivered **on demand** (order / user request workflow), then unpacked locally for the agriculture pipeline.

**Local layout (repository):** place the FIRMS GIS export under `INPUT/Proxy/ProxySpecific/Agriculture/VIIRS/`. The loader expects a VIIRS Shapefile Bundle matching `fire_archive_SV-C2_*.shp` (SV-C2: Suomi NPP + NOAA-20 combined, Collection 2) together with standard sidecars (`.dbf`, `.shx`, `.prj`, `.cpg`, …). Exactly **one** such `fire_archive_SV-C2_*.shp` must be present (first match alphabetically is used).

**Columns read by the pipeline:** `LATITUDE`, `LONGITUDE`, `ACQ_DATE`, `FRP`, `TYPE`, `CONFIDENCE`. Other attributes may be present in the download but are ignored unless re-exported under these names.

**Config pointer:** sector filepaths still use the YAML key `GFED4.folder` for this folder only (historic name).

**Access:** https://www.earthdata.nasa.gov/data/instruments/viirs/viirs-i-band-375-m-active-fire-data (instrument page; links to FIRMS / LANCE and related download paths).

**Access conditions:** NASA **Earthdata** login required for sanctioned bulk/ordered access; FIRMS distributes the GIS products used here.

#### C21: geospatial data from agricultural census

**Dataset:** Eurostat geospatial data from the agricultural census

**Description:** Experimental European gridded agricultural census data describing the spatial distribution of farm-related variables such as holdings, crop areas, irrigation, organic farming, and livestock. In this repository it is particularly useful for agriculture proxies that need livestock-related indicators such as cattle or sheep distributions.

**Method note:** the public product is released on a confidentiality-preserving multi-resolution grid rather than as raw farm locations.

**Access page:** https://ec.europa.eu/eurostat/web/experimental-statistics/geospatial-data-agricultural-census

**Dataset note:** the livestock layers are the most relevant ones for this project.

**Access conditions:** no account required.

### LUCAS land cover and land use

**Dataset:** EU LUCAS 2022

**Description:** LUCAS (Land Use/Cover Area frame Survey) is the in situ Eurostat survey describing land cover, land use, and field observations across Europe. It is useful when the proxy workflow needs point-based observations of actual land cover/use classes rather than purely remote-sensing-derived raster products.

**Typical use in this project:** supporting land-use interpretation, validation, or cross-checking for sector-specific spatial allocation.

**Access:** https://ec.europa.eu/eurostat/web/lucas/database

**Access conditions:** no account required for the database page and standard downloads.

### Other Combustion

#### Eurostat household energy consumption

**Dataset:** Eurostat household energy consumption statistics

**Description:** Statistical source describing household energy consumption patterns by end use and/or fuel at European and national level. In this project it is used to parameterise residential combustion proxies, especially when national totals must be distributed spatially with supporting geographic layers.

**Access:** https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Energy_consumption_in_households#Data_sources

**Access conditions:** no account required.

#### Hotmaps

**Dataset:** Hotmaps buildings / heat demand data

**Description:** Hotmaps provides European open datasets and tools for heating and cooling planning, including information related to building stock, gross floor area, and space-heating demand. In this repository it is used as a spatial indicator of residential and non-residential building energy demand for the `OtherCombustion` proxy.

**Main use here:** identifying where heat demand is concentrated, which is a useful proxy for small-scale stationary combustion in buildings.

**Access:** https://gitlab.com/hotmaps/buildings

**Additional background:** https://wiki.hotmaps.eu/en/Hotmaps-open-data-repositories

**Access conditions:** openly accessible project resources.

#### GAINS

**Dataset:** GAINS (Greenhouse Gas and Air Pollution Interactions and Synergies) model outputs

**Description:** GAINS provides country-level activity, technology, and emission-control information for air pollutant and greenhouse gas assessment. In this project, it is used to obtain the split of residential heating technologies and fuel types by country, which is then combined with spatial layers to build residential combustion proxies.

**Project-specific note:** use the advanced mode and retrieve data under `Activity data` / `Device types in the domestic sector`.

**Why it matters here:** GAINS helps translate a generic building-energy proxy into country-specific mixes of devices and fuels.

**Access:** https://gains.iiasa.ac.at/models/gains_versions.html

**Access conditions:** account required.

### Public Power

#### JRC Open Power Plants Database

**Dataset:** JRC Open Power Plants Database

**Description:** European power plant inventory compiled by the Joint Research Centre, designed to provide harmonised information on power generation facilities. In this repository it is used to locate and characterise public power plants for the `Public Power` proxy.

**Typical use:** facility-based spatial allocation of emissions or activity to known power generation sites.

**Access:** https://data.jrc.ec.europa.eu/dataset/9810feeb-f062-49cd-8e76-8d8cfd488a05

**Access conditions:** intended as open access; the JRC landing page may occasionally require retrying from a browser session.

### Shipping

#### EMODnet vessel density

**Dataset:** EMODnet Human Activities vessel density / route density layers

**Description:** European marine traffic density product showing the spatial distribution of vessel activity. In this project it is used as the main gridded proxy for shipping emissions, since it reflects the intensity of maritime traffic on shipping routes and in busy sea areas.

**Access path in viewer:** `EMODnet Human Activities` -> `Route Density` -> `Annual Totals` -> `2019-2025`

**Access:** https://emodnet.ec.europa.eu/geoviewer/

**Access conditions:** no account required.

#### Shipping lanes

**Dataset:** global shipping lanes from the `Shipping-Lanes` repository

**Description:** Vector dataset of major shipping routes. In this repository it is a complementary source to the EMODnet density rasters, useful for identifying the main navigational corridors when a simplified route geometry is needed.

**Access:** https://github.com/newzealandpaul/Shipping-Lanes/tree/main/data

**Access conditions:** no account required.

### Waste

#### UWWTD

**Dataset:** Urban Waste Water Treatment Directive reporting data

**Description:** European dataset on agglomerations and waste water treatment plants reported under the Urban Waste Water Treatment Directive. In this repository it is used for waste-sector proxy construction where waste water infrastructure and treatment capacity are relevant spatial indicators.

**Access:** https://www.eea.europa.eu/en/datahub/datahubitem-view/6244937d-1c2c-47f5-bdf1-33ca01ff1715

**Dataset note:** use the agglomerations and treatment plants tables/layers.

**Access conditions:** no account required.

### Imperviousness density

**Dataset:** Copernicus / WEkEO High Resolution Layer Imperviousness Density

**Description:** High-resolution Copernicus layer representing the spatial distribution of sealed or built-up surfaces across Europe. It is useful wherever the proxy depends on urbanisation intensity, built-up density, or impervious cover, and is particularly relevant for waste- and settlement-related spatial allocation.


**Documentation:** https://land.copernicus.eu/en/products/high-resolution-layer-imperviousness

**Access:** https://land.copernicus.eu/en/products/high-resolution-layer-imperviousness/imperviousness-density-2021#download

**Access condition** EU Login is required, dataset is free, you need to select the desired country

### Fugitive emissions

#### VIIRS Global Gas Flaring

**Dataset:** Global Gas Flaring Observed from Space, Earth Observation Group (Colorado School of Mines)

**Description:** Annual gas-flared volumes derived from VIIRS Nightfire (VNF) satellite observations, distributed as an Excel table with flare coordinates, radiant heat, and estimated flared volume per site. In this repository it is the primary spatial evidence for the gas flaring and residual losses group (G4) of the fugitive-emissions proxy, providing direct satellite-observed flare locations that are far more specific than infrastructure tags alone.

**Access:** https://eogdata.mines.edu/products/vnf/global_gas_flare.html

**Dataset note:** the 2020 annual product is used; coordinates are point-based and should be rasterised with a small buffer.

**Access conditions:** no account required.
### Fugitive

#### VIIRS Nightfire

**Dataset:** VIIRS Nightfire (VNF) flare detections

**Description:** Global satellite product derived from the Visible Infrared Imaging Radiometer Suite (VIIRS) . It is used as the primary spatial indicator for the gas flaring and residual losses group of the fugitive-emissions proxy, providing direct observational evidence of active flaring sites that is far more specific than infrastructure tags alone.

**Access:** https://eogdata.mines.edu/products/vnf/#download

**Dataset note:** use multi-year aggregated detections (mean radiant heat) to smooth episodic flaring activity.

**Access conditions:** free account required for bulk download.

#### Global Coal Mine Tracker

**Dataset:** Global Energy Monitor — Global Coal Mine Tracker (GCMT)

**Description:** Worldwide inventory of operating, proposed, and closed coal mines, including location, status, production capacity, and mine type. In this repository it is used as a complementary point/polygon source for the coal and solid fuels group of the fugitive-emissions proxy, supplementing CORINE mineral extraction polygons with explicit identification of operating coal mines.

**Access:** https://globalenergymonitor.org/projects/global-coal-mine-tracker/

**Access conditions:** free registration required to download the full dataset.

#### Global Oil & Gas Extraction Tracker

**Dataset:** Global Energy Monitor — Global Oil & Gas Extraction Tracker (GOGET)

**Description:** Worldwide inventory of oil and gas extraction projects, including upstream production sites, fields, and associated infrastructure with location, status, and production capacity. In this repository it is used as a spatial indicator for the oil upstream and transport group of the fugitive-emissions proxy, providing explicit upstream production sites that are otherwise sparsely tagged in OpenStreetMap.

**Access:** https://globalenergymonitor.org/projects/global-oil-gas-extraction-tracker/

**Access conditions:** free registration required to download the full dataset.
