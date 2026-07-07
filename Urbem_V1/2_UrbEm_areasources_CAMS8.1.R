# UrbEm — area sources module
# Annotated for readability and onboarding (2025-10)
# Original authors: M.O.P. Ramacher & A. Kakouri (2021)
# Update noted in header: CAMSv8.1 (Aug 20th 2025)
#
# Goal
# ----
# Downscale CAMS-REG v8.1 *area-source* emissions from regional (coarse) grid
# to an urban domain using *normalized proxies* (e.g., population, industry,
# shipping, agriculture...). The result is a set of sector-by-pollutant rasters
# aligned to the urban domain. Then we translate GNFR sectors to SNAP buckets
# and export an EPISODE‑CityChem (UECT) *area-source* CSV.
#
# If point sources have been processed first (using the point sources module),
# this script can use the *augmented* area sources that include unmatched point
# source emissions redistributed to area sources, ensuring total mass conservation.
#
# Workflow
# --------
# 1) Read CAMS v8.1 netCDF for selected year and GNFR sectors
# 2) Reproject/crop/resample to the urban domain
# 3) (Optional) Load augmented area sources from point source processing if available
#    (includes unmatched point source emissions added to area sources)
# 4) Prepare normalized per‑sector proxies on the domain grid (mass‑conserving
#    within coarse CAMS cells)
# 5) Distribute each sector's emissions with its sector‑specific proxy
# 6) (Traffic only) Optionally upweight urban core (GHSL)
# 7) Map GNFR → SNAP, build UECT area‑source records, and export CSV

####################
### INPUT section ###
####################

# Core libraries: rasters and netCDF IO
library(raster)  # gridded data handling
library(ncdf4)   # netCDF read support

# Helper scripts with reusable functions
# • proxy_preparation.R    : builds normalized proxy weights per coarse cell
#                            and applies proxy weights to emissions (proxy_cwd, proxy_distribution)
# • prepare_cams_*.R       : reads + rasterizes CAMS v8.1 netCDF by sector
setwd("C:/Users/leopi/PDM_local/")
source("./Urbem_V1/UrbEm_v1.0_Python_script/proxy_preparation.R")
source("./Urbem_V1/UrbEm_v1.0_Python_script/prepare_cams_v8.1_TNOftp_emissions.R")


# Paths to inputs -------------------------------------------------------------
# CAMS-REG-ANT v8.1 (TNO ftp) — all .nc files for the target year in one folder
emissions <- "C:/Users/leopi/PDM_local/INPUT/Emissions/"

# (Optional) Path to augmented area sources from point source processing
# If you ran the point sources module first, this file contains area sources that
# have been augmented with unmatched point source emissions (emissions in CAMS grid
# cells without matching RI-URBANS point sources). This ensures total mass conservation
# across point and area sources. Set to NULL to skip loading augmented sources.
#asrc_emissions_after_psrc_distribution <- NULL  # NULL to skip loading augmented sources

# Proxy rasters (GeoTIFF). Each will be normalized inside coarse CAMS cells.
population_proxy   <- "C:/Users/leopi/PDM_local/INPUT/Proxy/Population/GHS_POP_E2015_GLOBE_R2023A_4326_30ss_V1_0.tif"
industry_proxy     <- "C:/Users/leopi/PDM_local/INPUT/OLD/given_proxies/LU_Industry.tif"
wastefacility_proxy<- "C:/Users/leopi/PDM_local/INPUT/OLD/given_proxies/LU_Waste.tif"
agriculture_proxy  <- "C:/Users/leopi/PDM_local/INPUT/OLD/given_proxies/LU_Agriculture.tif"
shipping_proxy     <- "C:/Users/leopi/PDM_local/INPUT/OLD/given_proxies/LU_Shipping.tif"
aviation_proxy     <- "C:/Users/leopi/PDM_local/INPUT/OLD/given_proxies/LU_Airports.tif"
offroad_proxy      <- "C:/Users/leopi/PDM_local/INPUT/OLD/given_proxies/Non_Road_Mob_Sources.tif"
ghsl_urbancentre   <- "C:/Users/leopi/PDM_local/INPUT/OLD/given_proxies/ghs_europe_iso3.tif"

# Output location & filename stem --------------------------------------------
# site: "Milan" or "Cremona" (45 x 45 cells @ 1000 m — Comparison/Italy/domain.txt)
site <- "Cremona"

if (site == "Milan") {
  output       <- "C:/Users/leopi/PDM_local/OUTPUT/OLD/NEW_LOMBARDY"
  outputstring <- "Lombardy_areasources_CAMS-REG-AP_8.1"
  asrc_emissions_after_psrc_distribution <- "C:/Users/leopi/PDM_local/OUTPUT/OLD/NEW_LOMBARDY/Lombardy_pointsources_CAMS-REG-AP_8.1_corrected_areasources"
} else if (site == "Cremona") {
  output       <- "C:/Users/leopi/PDM_local/OUTPUT/OLD/Cremona_Lombardy"
  outputstring <- "Cremona_areasources_CAMS-REG-AP_8.1"
  asrc_emissions_after_psrc_distribution <- "C:/Users/leopi/PDM_local/OUTPUT/OLD/Cremona_Lombardy/Cremona_pointsources_CAMS-REG-AP_8.1_corrected_areasources"
} else {
  stop("Unknown site: ", site)
}

# Domain definition (target raster grid) -------------------------------------
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
# Full list including road transport (F1–F4)
GNFR <- c(
  "A_PublicPower", "B_Industry", "C_OtherStationaryComb",
  "D_Fugitives", "E_Solvents", "F1_RoadTransport_Exhaust_Gasoline",
  "F2_RoadTransport_Exhaust_Diesel", "F3_RoadTransport_Exhaust_LPG_gas",
  "F4_RoadTransport_NonExhaust", "G_Shipping", "H_Aviation",
  "I_OffRoad", "J_Waste", "K_AgriLivestock", "L_AgriOther"
)
# Alternative: GNFR without road transport (uncomment to use)
# GNFR <- c("A_PublicPower","B_Industry","C_OtherStationaryComb",
#           "D_Fugitives","E_Solvents","G_Shipping","H_Aviation",
#           "I_OffRoad","J_Waste","K_AgriLivestock","L_AgriOther")

####################
### END OF INPUT  ###
####################

# =============================================================================
# 1. Check if augmented area sources should be loaded
# =============================================================================

# Check if augmented_area_sources file exists and should be loaded
use_augmented_sources <- !is.null(asrc_emissions_after_psrc_distribution) &&
                         file.exists(asrc_emissions_after_psrc_distribution)

# =============================================================================
# 2. Read CAMS netCDF files (only if not using augmented sources, or for proxy prep)
# =============================================================================

# Fill the domain raster with 1s (handy carrier for extent/CRS)
domain <- setValues(domain, rep(1, ncell(domain)))

# Prepare output container for per‑sector rasters aligned to the domain
GNFR_raster <- list()

if (!use_augmented_sources) {
  # Full processing: load all CAMS data and process all sectors
  # Gather all files matching the chosen year
  df_nc <- list.files(path = emissions, pattern = paste0(year, ".nc"), full.names = TRUE)

  # Read & rasterize CAMS v8.1 for selected pollutants and sectors
  cams_europe <- prepare_cams_emissions(
    nc_file_path = df_nc,
    source_type  = "area",
    pollutants   = c("ch4","co","nh3","nmvoc","nox","pm10","pm2_5","sox"),
    sectors      = GNFR
  )


  # =============================================================================
  # 3. Project CAMS emissions to the urban domain
  # =============================================================================

  for (i in seq_along(GNFR)) {
    sector_name <- GNFR[i]

    # Get CAMS raster for this sector (WGS84)
    cams_wgs84 <- cams_europe$rasters[[i]]

    # Project to domain projection first (consistent with original approach)
    cams <- projectRaster(cams_wgs84, crs = crs(domain))

    # Crop to domain extent (with overlap at the boundaries)
    cams <- crop(cams, domain, snap = "out")

    # Keep the coarse first layer for later proxy normalization within coarse cells
    cams_origin <- cams[[1]]

    # Store cams_origin from first sector for proxy normalization (all sectors use same grid)
    if (i == 1) {
      cams_origin_for_proxy <- cams_origin
    }

    # Resample the full sector stack to the domain grid (nearest neighbor)
    GNFR_raster[[i]] <- resample(cams, domain, method = "ngb")
  }

  # Name each list entry with its GNFR sector for easier downstream handling
  names(GNFR_raster) <- GNFR

} else {
  # Using augmented sources: only load first sector to get cams_origin_for_proxy for proxy normalization
  # Gather all files matching the chosen year
  df_nc <- list.files(path = emissions, pattern = paste0(year, ".nc"), full.names = TRUE)

  # Read & rasterize CAMS v8.1 for first sector only (needed for proxy normalization)
  cams_europe <- prepare_cams_emissions(
    nc_file_path = df_nc,
    source_type  = "area",
    pollutants   = c("ch4","co","nh3","nmvoc","nox","pm10","pm2_5","sox"),
    sectors      = GNFR[1]  # Only first sector needed for cams_origin_for_proxy
  )

  # Process first sector to get cams_origin_for_proxy
  cams_wgs84 <- cams_europe$rasters[[1]]
  cams <- projectRaster(cams_wgs84, crs = crs(domain))
  cams <- crop(cams, domain, snap = "out")
  cams_origin_for_proxy <- cams[[1]]

  # =============================================================================
  # 3. Load augmented area sources
  # =============================================================================

  # Load the augmented area sources that include unmatched point source emissions
  load(asrc_emissions_after_psrc_distribution)
  # If augmented_area_sources exists, use it directly
  if (exists("augmented_area_sources")) {
    GNFR_raster <- augmented_area_sources
    # Ensure names match GNFR order (augmented_area_sources should already have correct names)
    names(GNFR_raster) <- GNFR
  } else {
    stop("augmented_area_sources not found in loaded file")
  }
}

# =============================================================================
# 4. PROXY preparation
# =============================================================================

# Build normalized proxy rasters on the domain grid for each *_proxy variable.
# Approach: for each proxy:
#  • project domain to proxy CRS
#  • crop & resample proxy to domain extent/resolution in its own CRS
#  • project proxy back to domain CRS
#  • compute per‑coarse‑cell normalized weights via proxy_cwd()
# The normalized weight arrays are then stored in variables without the _proxy
# suffix (e.g., population, industry, shipping, ...).

# Collect only the explicitly defined proxy variables from the INPUT section above
# This ensures we only get the intended proxy paths, not other variables that might contain "_proxy"
proxy_variable_names <- c("population_proxy", "industry_proxy", "wastefacility_proxy",
                          "agriculture_proxy", "shipping_proxy", "aviation_proxy",
                          "offroad_proxy")
proxies <- mget(proxy_variable_names, ifnotfound = list(NULL))
# Remove any NULL entries (in case a proxy variable wasn't defined)
proxies <- proxies[!sapply(proxies, is.null)]

for (i in seq_along(proxies)) {
  # If it's a character (file path), load it; otherwise use it directly
  if (is.character(proxies[[i]])) {
    proxy <- raster(proxies[[i]])
  } else {
    proxy <- proxies[[i]]
  }

  # Process in the proxy's native CRS to avoid edge data loss
  # Project domain to proxy projection, process and project back to domain
  # This leads to blurry proxies but makes sure no spatial data is "lost" due to projecting data
  domain_etrs <- projectRaster(domain, crs = crs(proxy))
  # Crop proxy to extent of our domain
  proxy <- crop(proxy, domain_etrs, snap = "out")
  # Bring it to our domain's resolution with bilinear sampling (in proxy CRS)
  proxy <- resample(proxy, domain_etrs, method = "bilinear")
  # Project proxy to our domain's projection and extent
  proxy <- projectRaster(proxy, domain, method = "ngb")

  # Normalize within each coarse CAMS cell using helper function
  # NOTE: cams_origin_for_proxy comes from the first sector (all sectors use same grid).
  # The function creates an ID raster internally to uniquely identify each coarse cell.
  norm_weights <- proxy_cwd(cams_origin_for_proxy, domain, proxy)

  # Save as a new variable named like the proxy without the trailing "_proxy"
  assign(substring(names(proxies)[i], 1, nchar(names(proxies)[i]) - 6), norm_weights)
}

# =============================================================================
# 5. Sector-by-sector proxy distribution
# =============================================================================

# For each GNFR sector, choose the appropriate proxy and top‑down distribute
# each pollutant layer accordingly (mass‑conserving inside coarse cells).
for (i in seq_along(GNFR)) {

  sector_name <- GNFR[i]

  #### A/B/D — Public power, Industry, Fugitives → Industry proxy
  if (sector_name == "A_PublicPower" ||
      sector_name == "B_Industry"   ||
      sector_name == "D_Fugitives") {
    GNFR_raster[[i]] <- proxy_distribution(
      emissions = GNFR_raster[[i]], proxy = industry,
      proxy_method = "coarse_cells_proxy"
    )
  }

  #### J — Waste → Waste‑facility proxy
  if (sector_name == "J_Waste") {
    GNFR_raster[[i]] <- proxy_distribution(
      emissions = GNFR_raster[[i]], proxy = wastefacility,
      proxy_method = "coarse_cells_proxy"
    )
  }

  #### C/E — Residential heating & Solvents → Population proxy
  if (sector_name == "C_OtherStationaryComb" ||
      sector_name == "E_Solvents") {
    GNFR_raster[[i]] <- proxy_distribution(
      emissions = GNFR_raster[[i]], proxy = population,
      proxy_method = "coarse_cells_proxy"
    )
  }

  #### F1–F4 — Road transport → Population proxy + GHSL upweight in urban core
  if (sector_name == "F1_RoadTransport_Exhaust_Gasoline" ||
      sector_name == "F2_RoadTransport_Exhaust_Diesel"   ||
      sector_name == "F3_RoadTransport_Exhaust_LPG_gas"  ||
      sector_name == "F4_RoadTransport_NonExhaust") {

    GNFR_raster[[i]] <- proxy_distribution(
      emissions = GNFR_raster[[i]], proxy = population,
      proxy_method = "coarse_cells_proxy"
    )

    # Optional: upweight traffic emissions in GHSL urban core (factor = 3)
    ghs <- raster(ghsl_urbancentre)
    domain_wgs <- projectRaster(domain, crs = crs(ghs))
    ghs_domain <- crop(ghs, domain_wgs)
    ghs_domain <- projectRaster(ghs_domain, domain)
    ghs_domain[ghs_domain > 0] <- 3
    ghs_domain[ghs_domain == 0] <- 1

    layernames <- names(GNFR_raster[[i]])
    GNFR_raster[[i]] <- ghs_domain * GNFR_raster[[i]]
    names(GNFR_raster[[i]]) <- layernames
  }

  #### G — Shipping → Shipping proxy
  if (sector_name == "G_Shipping") {
    GNFR_raster[[i]] <- proxy_distribution(
      emissions = GNFR_raster[[i]], proxy = shipping,
      proxy_method = "coarse_cells_proxy"
    )
  }

  #### H — Aviation → Airports proxy
  if (sector_name == "H_Aviation") {
    GNFR_raster[[i]] <- proxy_distribution(
      emissions = GNFR_raster[[i]], proxy = aviation,
      proxy_method = "coarse_cells_proxy"
    )
  }

  #### I — Non‑road mobility → Offroad proxy
  if (sector_name == "I_OffRoad") {
    GNFR_raster[[i]] <- proxy_distribution(
      emissions = GNFR_raster[[i]], proxy = offroad,
      proxy_method = "coarse_cells_proxy"
    )
  }

  #### K/L — Agriculture → Agriculture proxy
  if (sector_name == "K_AgriLivestock" || sector_name == "L_AgriOther") {
    GNFR_raster[[i]] <- proxy_distribution(
      emissions = GNFR_raster[[i]], proxy = agriculture,
      proxy_method = "coarse_cells_proxy"
    )
  }
}


# =============================================================================
# 6. Map GNFR → SNAP buckets
# =============================================================================

# (Sector split for SNAP3/SNAP4: here a fixed 80/20 split of GNFR B_Industry.)
SNAP1 <- GNFR_raster[["A_PublicPower"]]
SNAP2 <- GNFR_raster[["C_OtherStationaryComb"]]
SNAP3 <- GNFR_raster[["B_Industry"]] * 0.8
SNAP4 <- GNFR_raster[["B_Industry"]] * 0.2
SNAP5 <- GNFR_raster[["D_Fugitives"]]
SNAP6 <- GNFR_raster[["E_Solvents"]]
# Combine all road transport categories (F1–F4) into SNAP7
SNAP7 <- GNFR_raster[["F1_RoadTransport_Exhaust_Gasoline"]] +
  GNFR_raster[["F2_RoadTransport_Exhaust_Diesel"]] +
  GNFR_raster[["F3_RoadTransport_Exhaust_LPG_gas"]] +
  GNFR_raster[["F4_RoadTransport_NonExhaust"]]
SNAP8  <- GNFR_raster[["G_Shipping"]]
SNAP9  <- GNFR_raster[["J_Waste"]]
SNAP10 <- GNFR_raster[["K_AgriLivestock"]] + GNFR_raster[["L_AgriOther"]]
SNAP11 <- GNFR_raster[["H_Aviation"]] + GNFR_raster[["I_OffRoad"]]
SNAP11_Aviation <- GNFR_raster[["H_Aviation"]]
SNAP11_Offroad  <- GNFR_raster[["I_OffRoad"]]

# Bundle SNAP rasters into a list (plus aviation/offroad breakdown if needed)
snap_raster <- list(SNAP1,SNAP2,SNAP3,SNAP4,SNAP5,SNAP6,SNAP7,SNAP8,SNAP9,SNAP10,SNAP11,SNAP11_Aviation,SNAP11_Offroad)

# =============================================================================
# 7. Select SNAP sectors to export
# =============================================================================

# Examples:
# • all sectors:        snap_sectors <- c(1,2,3,4,5,6,7,8,9,10,11,12,13)
# • only aviation:      snap_sectors <- 11; snap_raster <- snap_raster[12]
# • here: all except traffic (since traffic lines come from a separate lsrc file)
snap_sectors <- c(1,2,3,4,5,6,8,9,10,11)
snap_raster  <- snap_raster[c(1,2,3,4,5,6,8,9,10,11)]

# =============================================================================
# 8. Format output for UECT (EPISODE-CityChem pre-processor)
# =============================================================================

# Build UECT area‑source table
# For each selected SNAP, we set an emission height (z) and create rectangles
# corresponding to each domain grid cell (south‑west to north‑east corners).
uect_area <- data.frame()

# SNAP sector mapping: snap_sectors index → (SNAP code, emission height in m)
snap_mapping <- list(
  `1`  = list(snap = 1,  z = 10),
  `2`  = list(snap = 2,  z = 10),
  `3`  = list(snap = 3,  z = 10),
  `4`  = list(snap = 4,  z = 10),
  `5`  = list(snap = 5,  z = 10),
  `6`  = list(snap = 6,  z = 0),
  `7`  = list(snap = 7,  z = 0),
  `8`  = list(snap = 8,  z = 10),
  `9`  = list(snap = 9,  z = 10),
  `10` = list(snap = 10, z = 0),
  `11` = list(snap = 11, z = 10)
)

for (i in seq_along(snap_sectors)) {
  # Get SNAP code and emission height from mapping
  mapping <- snap_mapping[[as.character(snap_sectors[[i]])]]
  snaps <- mapping$snap
  z <- mapping$z

  # Derive cell corners from domain cell centers
  sw <- coordinates(snap_raster[[i]]) - res(domain) / 2
  ne <- coordinates(snap_raster[[i]]) + res(domain) / 2

  # Pull pollutant layers; replace NAs with 0s later
  nox   <- values(snap_raster[[i]]$NOx)
  nmvoc <- values(snap_raster[[i]]$NMVOC)
  co    <- values(snap_raster[[i]]$CO)
  so2   <- values(snap_raster[[i]]$SO2)
  nh3   <- values(snap_raster[[i]]$NH3)
  pm25  <- values(snap_raster[[i]]$`PM2.5`)  # use backticks for layer names with dots
  pm10  <- values(snap_raster[[i]]$PM10)
  pollutants <- cbind(nox, nmvoc, co, so2, nh3, pm25, pm10)

  data <- cbind(snaps, sw, z, ne, z, pollutants)
  data[, 8:14][is.na(data[, 8:14])] <- 0  # replace NA pollutant values with 0
  colnames(data) <- c(
    "snap","xcor_sw","ycor_sw","zcor_sw",
    "xcor_ne","ycor_ne","zcor_ne",
    "NOx","NMVOC","CO","SO2","NH3","PM2.5","PM10"
  )

  uect_area <- rbind(uect_area, data)
}

# -----------------------------------------------------------------------------
# Remove rows with all-zero emissions
# -----------------------------------------------------------------------------

uect_input_csv <- NULL
for (i in seq_len(nrow(uect_area))) {
  if (sum(uect_area[i, c("NOx","NMVOC","CO","SO2","NH3","PM2.5","PM10")], na.rm = FALSE) > 0) {
    uect_input_csv <- rbind(uect_input_csv, uect_area[i, ])
  }
}

# =============================================================================
# 9. Write UECT area-source CSV
# =============================================================================

setwd(output)
write.csv(
  uect_input_csv,
  paste0(outputstring, "_", nrow(uect_input_csv), "_asrc.csv"),
  row.names = FALSE, quote = FALSE, na = "-999"
)

# =============================================================================
# 10. Clean environment
# =============================================================================

#rm(list = ls())
gc()
sum(uect_input_csv$PM2.5)/1000/1000
####################
### END OF SCRIPT ###
####################

# Notes & caveats -------------------------------------------------------------
# • Layer names with dots (e.g., PM2.5) require backticks in R: raster$`PM2.5`.
# • The GNFR→SNAP mapping shown mirrors the original script, including the 80/20
#   split of industry (B) into SNAP3/SNAP4 and a mapping of selected sectors to
#   default emission heights. Review these for your application.
# • The line that maps snap_sectors value 9 to snaps = 1 matches the original.
#   If you intended SNAP9, change to `snaps <- 9`.
# • The proxy_cwd() function now uses an ID raster approach to uniquely identify
#   each coarse CAMS cell, avoiding problems when many cells have the same value.
#   It only needs cams_origin (from first sector, all sectors use same grid) and domain.
