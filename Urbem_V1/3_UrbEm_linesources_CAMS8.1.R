# UrbEm — line sources module
# Annotated for readability and onboarding (2025-10)
# Original authors: M.O.P. Ramacher & A. Kakouri (2021)
# Update noted in header: CAMSv8.1 (Aug 20th 2025)
#
# Goal
# ----
# Convert CAMS-REG v8.1 *road-traffic* emissions from gridded area sources to
# per-road line emissions suitable for EPISODE‑CityChem (UECT). The script
# distributes area emissions onto OpenStreetMap (OSM) road networks using a
# VEIN-inspired allocation method, with optional population-based downscaling
# and urban-core upweighting. The result is a UECT-formatted CSV file with
# line sources (g/s) for road traffic emissions.
#
# Workflow
# --------
# 1) Read CAMS v8.1 netCDF emissions for selected traffic sectors (F1–F4) & pollutants
# 2) Reproject/crop/resample to the urban domain
# 3) (Optional) Downscale within coarse cells using population-density proxy (mass-conserving)
# 4) (Optional) Upweight emissions inside GHSL urban-core areas (e.g., ×3 for NOx)
# 5) Distribute area emissions onto OSM road networks (VEIN-inspired method)
# 6) Mass-correct line totals to match area totals (compensate for roadless cells)
# 7) Add road widths, compute line endpoints, convert to g/s, and write UECT CSV

####################
### INPUT section ###
####################

# Core libraries: rasters, vectors, OSM, and geometry operations
library(raster)   # gridded data handling
library(sf)       # spatial vector data
library(osmdata)  # OpenStreetMap data access
library(lwgeom)   # geometry helpers (start/end points, etc.)

# Helper scripts with reusable functions
# • areasources_to_osm_linesources.R : area→line allocation using OSM (VEIN-inspired)
# • proxy_preparation.R             : builds normalized proxy weights per coarse cell
# • prepare_cams_*.R                 : reads + rasterizes CAMS v8.1 netCDF by sector

setwd("C:/Users/leopi/PDM_local/")
source("./Urbem_V1/UrbEm_v1.0_Python_script/areasources_to_osm_linesources.R")
source("./Urbem_V1/UrbEm_v1.0_Python_script/proxy_preparation.R")
source("./Urbem_V1/UrbEm_v1.0_Python_script/prepare_cams_v8.1_TNOftp_emissions.R")

# Paths to inputs -------------------------------------------------------------
# CAMS-REG-ANT v8.1 (TNO ftp) — all .nc files for the target year in one folder
emissions <- "C:/Users/leopi/PDM_local/INPUT/Emissions/"

# Proxy datasets (GeoTIFF). Population density for within-cell downscaling.
population_proxy   <- "C:/Users/leopi/PDM_local/INPUT/Proxy/Population/GHS_POP_E2015_GLOBE_R2023A_4326_30ss_V1_0.tif"
ghsl_urbancentre   <- "C:/Users/leopi/PDM_local/INPUT/OLD/given_proxies/ghs_europe_iso3.tif"

# Output location & filename stem --------------------------------------------
# site: "Milan" or "Cremona" (45 x 45 cells @ 1000 m — Comparison/Italy/domain.txt)
site <- "Cremona"

if (site == "Milan") {
  output       <- "C:/Users/leopi/PDM_local/OUTPUT/OLD/NEW_LOMBARDY"
  outputstring <- "Lombardy_linesources_v3_CAMS-REG-AP_v8.1_2021"
} else if (site == "Cremona") {
  output       <- "C:/Users/leopi/PDM_local/OUTPUT/OLD/Cremona_Lombardy"
  outputstring <- "Cremona_linesources_v3_CAMS-REG-AP_v8.1_2021"
} else {
  stop("Unknown site: ", site)
}
osm_cache    <- paste0(output, "/osm_roads_cache.gpkg")

# Domain definition (target raster grid) -------------------------------------
# Athens, Greece (UTM zone 34N). Update for other cities as needed.
#domain <- raster(nrow = 45, ncol = 45,
#                 ymn = 4191261, xmn = 716397,
#                 ymx = 4236261, xmx = 761397,
#                 crs = "+proj=utm +zone=34 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

# Lombardy sub-domains (EPSG:32632) — cell corners = centre +/- 500 m
if (site == "Milan") {
  domain <- raster(nrow = 45, ncol = 45,
                   ymn = 5013000, xmn = 493000,
                   ymx = 5058000, xmx = 538000,
                   crs = "+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
} else if (site == "Cremona") {
  domain <- raster(nrow = 45, ncol = 45,
                   ymn = 4976000, xmn = 557000,
                   ymx = 5021000, xmx = 602000,
                   crs = "+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
}
# Inventory year to process
year <- 2021

# GNFR sectors to include (CAMS v8.1) ----------------------------------------
# Traffic only: F1–F4 exhaust/non-exhaust and fuel splits
GNFR <- c("F1_RoadTransport_Exhaust_Gasoline",
          "F2_RoadTransport_Exhaust_Diesel",
          "F3_RoadTransport_Exhaust_LPG_gas",
          "F4_RoadTransport_NonExhaust")

# Processing options ---------------------------------------------------------
# Choose whether to apply a within-cell proxy before distributing to lines
# "yes" = downscale with population density; "no" = uniform within coarse cells
pop_proxy <- "yes"  # recommended: "yes"

# Optionally inflate emissions in the GHSL urban centre (Kuik et al. 2019)
centre <- "yes"          # "yes"/"no"
centre_factor <- 1        # e.g., 3× for NOx as in literature
pollutants <- "all"      # "all" or "nox" (what gets upweighted)

####################
### END OF INPUT  ###
####################

# =============================================================================
# 1. Read CAMS netCDF files
# =============================================================================

# Gather all files matching the chosen year
cams_file <- list.files(path = emissions, pattern = paste0(year, ".nc"), full.names = TRUE)

# Read & rasterize CAMS v8.1 for selected pollutants and sectors
cams_europe <- prepare_cams_emissions(
  nc_file_path = cams_file,
  source_type  = "area",
  pollutants   = c("ch4", "co", "nh3", "nmvoc", "nox", "pm10", "pm2_5", "sox"),
  sectors      = GNFR
)

# Quick visual check (example: sector F2, NOx)
#plot(cams_europe$rasters[["F2"]][["NOx"]])
#sum(getValues(cams_europe$rasters[["F2"]][["NOx"]]))

# =============================================================================
# 2. Project CAMS emissions to the urban domain
# =============================================================================

# Fill the domain raster with 1s (handy carrier for extent/CRS)
domain <- setValues(domain, rep(1, ncell(domain)))

# Prepare output container for per‑sector rasters aligned to the domain
GNFR_raster <- list()

for (i in seq_along(GNFR)) {
  # Reproject from CAMS CRS to the domain CRS
  cams <- projectRaster(cams_europe$rasters[[i]], crs = crs(domain))
  # Crop to domain extent (snapping outward so we keep edge information)
  cams <- crop(cams, domain, snap = "out")

  # Keep the coarse first layer for later proxy normalization within coarse cells
  cams_origin <- cams[[1]]
  # Prepare a domain‑grid version of a single layer for proxy helper functions
  cams_domain <- resample(cams[[1]], domain, method = "ngb")

  # Resample the full sector stack to the domain grid (nearest neighbor)
  GNFR_raster[[i]] <- resample(cams, domain, method = "ngb")
}

# Name each list entry with its GNFR sector for easier downstream handling
names(GNFR_raster) <- GNFR

# Sum the four traffic sectors to a single multi-layer stack (one layer per pollutant)
cams <- GNFR_raster[[1]] + GNFR_raster[[2]] + GNFR_raster[[3]] + GNFR_raster[[4]]
# plot(cams$NOx)

# =============================================================================
# 3. Apply population-based proxy downscaling (optional)
# =============================================================================

if (pop_proxy == "yes") {
  # Read population density (GHS) and align it with our domain carefully
  pop <- raster(population_proxy)

  # Strategy: project the *domain* to the proxy CRS, process there, then project back.
  # This avoids spatial gaps at edges that can occur when directly warping the proxy.
  options(warn = -1)
  domain_etrs <- projectRaster(domain, crs = crs(pop))     # domain in proxy CRS
  pop <- crop(pop, domain_etrs, snap = "out")              # clip to domain bbox
  pop <- resample(pop, domain_etrs, method = "bilinear")   # match proxy resolution
  pop <- projectRaster(pop, domain, method = "ngb")        # back to domain grid

  # Build normalized weights inside each coarse CAMS cell using the helper
  # The idea: for each fine cell in a coarse cell, compute pop / sum(pop_in_coarse_cell)
  # pop_norm <- proxy_cwd(cams, cams_origin, pop)
  pop_norm <- proxy_cwd(domain = cams, cams_origin = cams_origin, proxy = pop,
                            sparse_threshold = 0.05, max_weight_per_cell = 0.5)

  # Multiply each pollutant layer by the normalized population weights (mass-conserving)
  temp <- vector("list", nlayers(cams))
  for (j in seq_len(nlayers(cams))) {
    temp[[j]] <- cams[[j]] * pop_norm
    names(temp[[j]]) <- names(cams[[j]])
  }
  cams <- brick(temp)
}
# plot(cams)

# =============================================================================
# 4. Apply GHSL urban-core upweighting (optional)
# =============================================================================

if (centre == "yes") {
  # Report pre-weighting NOx total in kt/yr
  message(paste0("Total NOx before urban-centre weighting: ",
                 sum(getValues(cams$NOx), na.rm = TRUE) / 1e6, " kt NOx"))

  # Prepare GHSL urban-centre mask on domain grid
  ghs <- raster(ghsl_urbancentre)
  domain_wgs <- projectRaster(domain, crs = crs(ghs))
  ghs_domain <- crop(ghs, domain_wgs)
  ghs_domain <- projectRaster(ghs_domain, domain)

  # Convert to multiplicative mask: centre_factor inside core, 1 elsewhere
  ghs_domain[ghs_domain > 0] <- centre_factor
  ghs_domain[ghs_domain == 0] <- 1

  # Apply weighting either to all pollutants or NOx only
  if (pollutants == "all") {
    layernames <- names(cams)
    cams <- ghs_domain * cams
    names(cams) <- layernames
  } else if (pollutants == "nox") {
    cams$NOx <- cams$NOx * ghs_domain
  }

  message(paste0(
    "Total NOx after urban-centre weighting (×", centre_factor, "): ",
    sum(getValues(cams$NOx), na.rm = TRUE) / 1e6, " kt NOx")
  )
}
# plot(cams$NOx)

# =============================================================================
# 5. Distribute area emissions to OSM line sources
# =============================================================================

# Uses VEIN-inspired allocation (Ibarra et al.), customized in helper script.
# This function allocates gridded area emissions to OSM road network segments.
out <- area_to_osm_lines(domain = domain, area_emissions = cams, osm_cache = osm_cache)
out <- na.omit(out)

# -----------------------------------------------------------------------------
# Mass correction
# -----------------------------------------------------------------------------

# Ensure the sum over all line segments equals the original area-sum per pollutant
# (compensate for roadless cells during downscaling)
out$NOx   <- out$NOx   * sum(getValues(cams$NOx),   na.rm = TRUE) / sum(out$NOx,   na.rm = FALSE)
out$NMVOC <- out$NMVOC * sum(getValues(cams$NMVOC), na.rm = TRUE) / sum(out$NMVOC, na.rm = FALSE)
out$CO    <- out$CO    * sum(getValues(cams$CO),    na.rm = TRUE) / sum(out$CO,    na.rm = FALSE)
out$SO2   <- out$SO2   * sum(getValues(cams$SO2),   na.rm = TRUE) / sum(out$SO2,   na.rm = FALSE)
out$NH3   <- out$NH3   * sum(getValues(cams$NH3),   na.rm = TRUE) / sum(out$NH3,   na.rm = FALSE)
out$PM25  <- out$PM25  * sum(getValues(cams$`PM2.5`), na.rm = TRUE) / sum(out$PM25,  na.rm = FALSE)
out$PM10  <- out$PM10  * sum(getValues(cams$PM10),   na.rm = TRUE) / sum(out$PM10,  na.rm = FALSE)

# Quick mass sanity check (kt/yr)
# sum(out$NOx, na.rm = TRUE) / 1e6
# sum(getValues(cams$NOx), na.rm = TRUE) / 1e6

# =============================================================================
# 6. Assign road widths from OSM road class
# =============================================================================

# Map OSM roadtype to an approximate physical width (meters)
# (Adjust values to your local defaults as needed)
unique(out$roadtype)
out$roadtype[out$roadtype == "motorway"]       <- 20
out$roadtype[out$roadtype == "motorway_link"]  <- 20
out$roadtype[out$roadtype == "trunk"]          <- 16
out$roadtype[out$roadtype == "trunk_link"]     <- 16
out$roadtype[out$roadtype == "primary"]        <- 12
out$roadtype[out$roadtype == "primary_link"]   <- 12
out$roadtype[out$roadtype == "secondary"]      <- 12
out$roadtype[out$roadtype == "secondary_link"] <- 12
out$roadtype[out$roadtype == "tertiary"]       <- 8
out$roadtype[out$roadtype == "tertiary_link"]  <- 8

# Rename columns to final schema and keep geometry
y <- st_geometry(out)
out$width <- as.numeric(out$roadtype)

# Rebuild an sf object with tidy names in the intended order
out <- st_sf(
  NOx    = out$NOx,
  NMVOC  = out$NMVOC,
  CO     = out$CO,
  SO2    = out$SO2,
  NH3    = out$NH3,
  PM25   = out$PM25,
  PM10   = out$PM10,
  width  = out$width,
  geometry = y
)

# =============================================================================
# 7. Compute line start/end coordinates
# =============================================================================

# UECT requires line endpoints (start and end coordinates)
start <- st_startpoint(out$geometry)
start <- unlist(st_geometry(start)) |> matrix(ncol = 2, byrow = TRUE) |> data.frame() |> setNames(c("x_start", "y_start"))

end <- st_endpoint(out$geometry)
end <- unlist(st_geometry(end))   |> matrix(ncol = 2, byrow = TRUE) |> data.frame() |> setNames(c("x_end", "y_end"))

# =============================================================================
# 8. Format output for UECT (EPISODE-CityChem pre-processor)
# =============================================================================

# Prepare tabular output in UECT format
snap      <- 7                                  # SNAP sector code for traffic
xcor_start <- as.integer(start$x_start)
ycor_start <- as.integer(start$y_start)
xcor_end   <- as.integer(end$x_end)
ycor_end   <- as.integer(end$y_end)
elevation  <- 0
width      <- out$width

# Unit conversion: kg/year → g/s
kgyr_to_gs <- function(x) x * 1e3 / (365 * 24 * 3600)
NOx   <- kgyr_to_gs(out$NOx)
NMVOC <- kgyr_to_gs(out$NMVOC)
CO    <- kgyr_to_gs(out$CO)
SO2   <- kgyr_to_gs(out$SO2)
NH3   <- kgyr_to_gs(out$NH3)
PM25  <- kgyr_to_gs(out$PM25)
PM10  <- kgyr_to_gs(out$PM10)

# Assemble the UECT line table (as.data.frame to drop sf geometry for CSV)
uect_lines <- as.data.frame(cbind(
  snap, xcor_start, ycor_start, xcor_end, ycor_end, elevation, width,
  NOx, NMVOC, CO, SO2, NH3, PM25, PM10
))
colnames(uect_lines) <- c(
  "snap", "xcor_start", "ycor_star", "xcor_end", "ycor_end", "elevation",
  "width", "NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10"
)

# Add PN placeholder for CityChem 1.5 compatibility
uect_lines$PN <- NA

# head(uect_lines)

# =============================================================================
# 9. Write UECT line-source CSV
# =============================================================================

setwd(output)
write.csv(
  uect_lines,
  paste0(outputstring, "_", nrow(uect_lines), "_lsrc.csv"),
  row.names = FALSE, quote = FALSE, na = "-999"
)

# =============================================================================
# 10. Clean environment
# =============================================================================

#rm(list = ls())
gc()

####################
### END OF SCRIPT ###
####################

# Notes & caveats -------------------------------------------------------------
# • If you target a different city: update the domain raster extent/CRS and the
#   outputstring. Keep CRS consistent across steps.
# • If you set pop_proxy = "no", emissions are uniform within coarse cells before
#   line allocation. Mass conservation is still enforced after line allocation.
# • The GHSL centre upweighting can be limited to NOx by setting pollutants = "nox".
# • The road width mapping is a simple heuristic – tailor to your network if needed.
# • Layer names with dots (e.g., PM2.5) require backticks in R: raster$`PM2.5`.
