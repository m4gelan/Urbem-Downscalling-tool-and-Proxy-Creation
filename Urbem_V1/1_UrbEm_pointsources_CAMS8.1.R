# UrbEm — point sources module
# Annotated for readability and onboarding (2025-10)
# Original authors: M.O.P. Ramacher & A. Kakouri (2021)
# Update noted in header: CAMSv8.1 (Aug 20th 2025)
#
# Goal
# ----
# Process CAMS-REG-ANT v8.1 *point-source* emissions by distributing them from
# coarse grid cells to real point source locations from the RI-URBANS inventory.
# Emissions in CAMS grid cells without matching RI-URBANS points are added back
# to the corresponding area-source rasters to preserve total mass. The result is
# a UECT-formatted CSV file for EPISODE-CityChem point-source input.
#
# Workflow
# --------
# 1) Read CAMS v8.1 point source emissions (gridded with cell center coordinates)
# 2) Read RI-URBANS CSV file (point sources with real coordinates)
# 3) Crop both datasets to the defined urban domain
# 4) For each sector and pollutant:
#    • Identify CAMS grid cells containing RI-URBANS point sources
#    • Distribute CAMS emissions proportionally to RI-URBANS points based on
#      their emission weights (sum of RI-URBANS emissions per cell)
#    • Mark CAMS emissions without matching RI-URBANS points as "unmatched"
# 5) Add unmatched point source emissions to corresponding CAMS area sources
# 6) Aggregate distributed points by location and create UECT-formatted CSV

####################
### INPUT section ###
####################

# Core libraries: rasters and spatial data handling
library(raster)  # gridded data handling
library(sf)      # spatial vector data
library(reshape2) # data reshaping (dcast)

# Helper scripts with reusable functions
# • prepare_cams_*.R : reads + rasterizes CAMS v8.1 netCDF by sector
setwd("C:/Users/leopi/PDM_local/")
source("./UrbEm_V1/UrbEm_v1.0_Python_script/prepare_cams_v8.1_TNOftp_emissions.R")

# Paths to inputs -------------------------------------------------------------
# CAMS-REG-ANT v8.1 (TNO ftp) — all .nc files for the target year in one folder
emissions <- "C:/Users/leopi/PDM_local/INPUT/Emissions/"

# RI-URBANS point source inventory (CSV with real coordinates)
RIURBANS_emissions <- "C:/Users/leopi/PDM_local/INPUT/Proxy/Emissions_RI-URBANS_2018_main_v1_0.csv"

# Output location & filename stem --------------------------------------------
# site: "Milan" or "Cremona" (45 x 45 cells @ 1000 m — Comparison/Italy/domain.txt)
site <- "Cremona"

if (site == "Milan") {
  output       <- "C:/Users/leopi/PDM_local/OUTPUT/OLD/NEW_LOMBARDY"
  outputstring <- "Lombardy_pointsources_CAMS-REG-AP_8.1"
} else if (site == "Cremona") {
  output       <- "C:/Users/leopi/PDM_local/OUTPUT/OLD/Cremona_Lombardy"
  outputstring <- "Cremona_pointsources_CAMS-REG-AP_8.1"
} else {
  stop("Unknown site: ", site)
}

# Domain definition (target raster grid) -------------------------------------
# Hamburg, Germany (UTM zone 32N). Update for other cities as needed.
#domain <- raster(nrow = 50, ncol = 50,
#                 ymn = 5908656, xmn = 536750,
#                 ymx = 5958656, xmx = 586750,
#                 crs = "+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

# Athens, Greece (UTM zone 34N)
#domain <- raster(nrow = 45, ncol = 45,
#                 ymn = 4191261, xmn = 716397,
#                 ymx = 4236261, xmx = 761397,
#                 crs = "+proj=utm +zone=34 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

# Ioannina, Greece (UTM zone 34N)
# domain <- raster(nrow = 30, ncol = 30,
#                  ymn = 4375636, xmn = 468812,
#                  ymx = 4405636, xmx = 498812,
#                  crs = "+proj=utm +zone=34 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

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
# Fill the domain raster with 1s (handy carrier for extent/CRS)
domain <- setValues(domain, rep(1, ncell(domain)))

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

# create geotiff of Ioannina
#writeRaster(domain, "domain.tif", format = "GTiff", overwrite = TRUE)

####################
### END OF INPUT  ###
####################

# =============================================================================
# 1. Read CAMS netCDF files
# =============================================================================

# Gather all files matching the chosen year
df_nc <- list.files(path = emissions, pattern = paste0(year, ".nc"), full.names = TRUE)

# Read & rasterize CAMS v8.1 for selected pollutants and sectors --> POINT SOURCES ONLY
cams_europe_psrc <- prepare_cams_emissions(
  nc_file_path = df_nc,
  source_type  = "point",
  pollutants   = c("ch4","co","nh3","nmvoc","nox","pm10","pm2_5","sox"),
  sectors      = GNFR
)

# Read & rasterize CAMS v8.1 for selected pollutants and sectors --> AREA SOURCES ONLY
# (needed for adding unmatched point source emissions later)
cams_europe_asrc <- prepare_cams_emissions(
  nc_file_path = df_nc,
  source_type  = "area",
  pollutants   = c("ch4","co","nh3","nmvoc","nox","pm10","pm2_5","sox"),
  sectors      = GNFR
)

# =============================================================================
# 2. Read CAMS RI-URBANS CSV file
# =============================================================================

# Read RI-URBANS point source inventory (contains real coordinates)
cams_RIURBANS <- read.csv(RIURBANS_emissions, sep = ";", dec = ".")
# Select only point sources
cams_RIURBANS <- subset(cams_RIURBANS, SourceType == "P")

# =============================================================================
# 3. Extract CAMS grid information for point sources
# =============================================================================

# Get grid cell centers and resolution from CAMS point source data
cams_grid_lons <- cams_europe_psrc$grid_info$lons
cams_grid_lats <- cams_europe_psrc$grid_info$lats
cams_res_lon <- diff(cams_grid_lons)[1]  # grid cell width in degrees
cams_res_lat <- diff(cams_grid_lats)[1]  # grid cell height in degrees

# Calculate grid cell boundaries (for spatial matching)
# Each grid cell center at (lon[i], lat[j]) represents a cell with boundaries:
# lon: [lon[i] - res_lon/2, lon[i] + res_lon/2]
# lat: [lat[j] - res_lat/2, lat[j] + res_lat/2]

# =============================================================================
# 4. Filter data to domain extent
# =============================================================================

# Project domain to WGS84 for spatial filtering
domain_wgs <- projectRaster(domain, crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")


# Filter RI-URBANS data to domain extent
cams_RIURBANS_domain <- subset(cams_RIURBANS,
                               Lon >= xmin(domain_wgs) & Lon <= xmax(domain_wgs) &
                               Lat >= ymin(domain_wgs) & Lat <= ymax(domain_wgs))

# Extract GNFR code from GNFR_Sector (already single letter, but ensure consistency)
cams_RIURBANS_domain$GNFR_code <- as.character(cams_RIURBANS_domain$GNFR_Sector)

# Map GNFR full names to codes for matching
GNFR_codes <- sapply(strsplit(GNFR, "_"), `[[`, 1)
names(GNFR_codes) <- GNFR

# =============================================================================
# 5. Define pollutant mappings
# =============================================================================

# Map pollutant names between CAMS (NetCDF variables), RI-URBANS (CSV columns), and UECT (output)
# CAMS pollutants: ch4, co, nh3, nmvoc, nox, pm10, pm2_5, sox
# RI-URBANS pollutants: CH4, CO, NH3, NMVOC, NOX, PM10, PM2_5, SO2
pollutant_map <- data.frame(
  cams = c("ch4", "co", "nh3", "nmvoc", "nox", "pm10", "pm2_5", "sox"),
  riurbans = c("CH4", "CO", "NH3", "NMVOC", "NOX", "PM10", "PM2_5", "SO2"),
  uect = c("CH4", "CO", "NH3", "NMVOC", "NOx", "PM10", "PM2.5", "SO2"),
  stringsAsFactors = FALSE
)

# Map CAMS NetCDF variable names to raster layer names (pretty names)
layer_name_map <- c(
  "ch4" = "CH4",
  "co" = "CO",
  "nh3" = "NH3",
  "nmvoc" = "NMVOC",
  "nox" = "NOx",
  "pm10" = "PM10",
  "pm2_5" = "PM2.5",
  "sox" = "SO2"
)

# =============================================================================
# 6. Extract CAMS point source emissions per grid cell
# =============================================================================

# We need to extract emissions from CAMS rasters before they're projected/resampled
# The rasters are in WGS84 lat/lon with grid cell centers at cams_grid_lons/lats
# Store as: sector -> pollutant -> grid_cell -> emission_value
cams_psrc_emissions <- list()
unmatched_emissions <- list()  # Track emissions not distributed to RI-URBANS

for (sector_idx in seq_along(GNFR)) {
  sector_name <- GNFR[sector_idx]
  sector_code <- GNFR_codes[sector_name]

  # Get CAMS raster for this sector (WGS84, not projected)
  cams_sector_raster <- cams_europe_psrc$rasters[[sector_code]]

  if (is.null(cams_sector_raster)) {
    next
  }

  # Initialize storage for this sector
  cams_psrc_emissions[[sector_name]] <- list()
  unmatched_emissions[[sector_name]] <- list()

  # Extract emissions for each pollutant
  for (poll_idx in seq_len(nrow(pollutant_map))) {
    cams_poll <- pollutant_map$cams[poll_idx]
    layer_name <- layer_name_map[cams_poll]

    # Get pollutant layer from raster stack
    poll_raster <- NULL
    if (!is.null(layer_name) && layer_name %in% names(cams_sector_raster)) {
      poll_raster <- cams_sector_raster[[layer_name]]
    } else if (cams_poll %in% names(cams_sector_raster)) {
      poll_raster <- cams_sector_raster[[cams_poll]]
    }

    if (!is.null(poll_raster)) {
      # Extract all non-zero grid cells
      poll_values <- getValues(poll_raster)
      poll_coords <- coordinates(poll_raster)

      # Store emissions per grid cell
      grid_emissions <- data.frame(
        lon = poll_coords[,1],
        lat = poll_coords[,2],
        emission = poll_values,
        stringsAsFactors = FALSE
      )

      # Filter to domain extent and non-zero emissions
      grid_emissions <- subset(grid_emissions,
                              lon >= xmin(domain_wgs) & lon <= xmax(domain_wgs) &
                              lat >= ymin(domain_wgs) & lat <= ymax(domain_wgs) &
                              emission > 0)

      if (nrow(grid_emissions) > 0) {
        cams_psrc_emissions[[sector_name]][[cams_poll]] <- grid_emissions
      }
    }
  }
}

# =============================================================================
# 7. Match RI-URBANS points to CAMS grid cells and distribute emissions
# =============================================================================

# Distribution method:
#   For every CAMS point-source grid cell (sector/pollutant), find all RI-URBANS
#   point sources (real coordinates) that fall inside that grid cell.
#   If several RI-URBANS points share the same CAMS cell, distribute the CAMS
#   emission proportionally to their RI-URBANS-reported emissions for that pollutant.
#   If no RI-URBANS points exist inside the cell, the CAMS emission is stored as
#   "unmatched" and later added back to the corresponding area-source raster so the
#   total sector/pollutant mass is preserved.

distributed_points <- list()

for (sector_idx in seq_along(GNFR)) {
  sector_name <- GNFR[sector_idx]
  sector_code <- GNFR_codes[sector_name]

  # Filter RI-URBANS data for this sector
  riurbans_sector <- subset(cams_RIURBANS_domain, GNFR_Sector == sector_code)

  # Process each pollutant
  for (poll_idx in seq_len(nrow(pollutant_map))) {
    cams_poll <- pollutant_map$cams[poll_idx]
    riurbans_poll <- pollutant_map$riurbans[poll_idx]

    # Get CAMS emissions for this sector/pollutant
    if (is.null(cams_psrc_emissions[[sector_name]][[cams_poll]])) {
      next
    }

    cams_grid_emissions <- cams_psrc_emissions[[sector_name]][[cams_poll]]
    distributed_emissions <- data.frame()
    unmatched_grid_cells <- data.frame()

    # If no RI-URBANS points for this sector, mark all CAMS emissions as unmatched
    if (nrow(riurbans_sector) == 0) {
      for (grid_idx in seq_len(nrow(cams_grid_emissions))) {
        unmatched_grid_cells <- rbind(unmatched_grid_cells,
          data.frame(
            sector = sector_code,
            lon = cams_grid_emissions$lon[grid_idx],
            lat = cams_grid_emissions$lat[grid_idx],
            pollutant = cams_poll,
            emission = cams_grid_emissions$emission[grid_idx],
            stringsAsFactors = FALSE
          ))
      }
    } else {
      # Normal matching process when RI-URBANS points exist
      for (grid_idx in seq_len(nrow(cams_grid_emissions))) {
        grid_lon <- cams_grid_emissions$lon[grid_idx]
        grid_lat <- cams_grid_emissions$lat[grid_idx]
        cams_emission <- cams_grid_emissions$emission[grid_idx]

        # Find RI-URBANS points within this grid cell
        # Grid cell boundaries: [lon - res_lon/2, lon + res_lon/2] x [lat - res_lat/2, lat + res_lat/2]
        riurbans_in_cell <- subset(riurbans_sector,
                                  Lon >= (grid_lon - cams_res_lon/2) &
                                  Lon <= (grid_lon + cams_res_lon/2) &
                                  Lat >= (grid_lat - cams_res_lat/2) &
                                  Lat <= (grid_lat + cams_res_lat/2))

        if (nrow(riurbans_in_cell) > 0) {
          # Calculate total RI-URBANS emissions for this pollutant in this cell
          riurbans_emissions <- riurbans_in_cell[[riurbans_poll]]
          riurbans_emissions[is.na(riurbans_emissions)] <- 0
          total_riurbans_emission <- sum(riurbans_emissions, na.rm = TRUE)

          if (total_riurbans_emission > 0) {
            # Distribute CAMS emissions proportionally based on RI-URBANS emissions
            weights <- riurbans_emissions / total_riurbans_emission
            distributed <- cams_emission * weights

            # Project RI-URBANS coordinates to domain CRS
            pts_sf <- st_as_sf(riurbans_in_cell, coords = c("Lon", "Lat"),
                              crs = "+proj=longlat +datum=WGS84 +no_defs")
            pts_proj <- st_transform(pts_sf, crs = crs(domain))
            coords_proj <- st_coordinates(pts_proj)

            # Create output records for each RI-URBANS point
            for (pt_idx in seq_len(nrow(riurbans_in_cell))) {
              distributed_emissions <- rbind(distributed_emissions,
                data.frame(
                  sector = sector_code,
                  xcor = coords_proj[pt_idx, 1],
                  ycor = coords_proj[pt_idx, 2],
                  lon = riurbans_in_cell$Lon[pt_idx],
                  lat = riurbans_in_cell$Lat[pt_idx],
                  pollutant = cams_poll,
                  emission = distributed[pt_idx],
                  stringsAsFactors = FALSE
                ))
            }
          } else {
            # No RI-URBANS emissions in this cell, mark as unmatched
            unmatched_grid_cells <- rbind(unmatched_grid_cells,
              data.frame(
                sector = sector_code,
                lon = grid_lon,
                lat = grid_lat,
                pollutant = cams_poll,
                emission = cams_emission,
                stringsAsFactors = FALSE
              ))
          }
        } else {
          # No RI-URBANS points in this grid cell, mark as unmatched
          unmatched_grid_cells <- rbind(unmatched_grid_cells,
            data.frame(
              sector = sector_code,
              lon = grid_lon,
              lat = grid_lat,
              pollutant = cams_poll,
              emission = cams_emission,
              stringsAsFactors = FALSE
            ))
        }
      }
    }

    # Store distributed emissions for this sector/pollutant
    if (nrow(distributed_emissions) > 0) {
      if (is.null(distributed_points[[sector_name]])) {
        distributed_points[[sector_name]] <- list()
      }
      distributed_points[[sector_name]][[cams_poll]] <- distributed_emissions
    }

    # Store unmatched emissions
    if (nrow(unmatched_grid_cells) > 0) {
      unmatched_emissions[[sector_name]][[cams_poll]] <- unmatched_grid_cells
    }
  }
}

# =============================================================================
# 8. Validation: Check emission balance
# =============================================================================

# Check that CAMS emissions = distributed to points + unmatched as area (for each sector/pollutant)
validation_results <- data.frame()

for (sector_name in names(cams_psrc_emissions)) {
  sector_code <- GNFR_codes[sector_name]

  for (poll_name in names(cams_psrc_emissions[[sector_name]])) {
    # Get original CAMS emissions
    cams_total <- sum(cams_psrc_emissions[[sector_name]][[poll_name]]$emission, na.rm = TRUE)

    # Get distributed emissions
    distributed_total <- 0
    if (!is.null(distributed_points[[sector_name]][[poll_name]])) {
      distributed_total <- sum(distributed_points[[sector_name]][[poll_name]]$emission, na.rm = TRUE)
    }

    # Get unmatched emissions
    unmatched_total <- 0
    if (!is.null(unmatched_emissions[[sector_name]][[poll_name]])) {
      unmatched_total <- sum(unmatched_emissions[[sector_name]][[poll_name]]$emission, na.rm = TRUE)
    }

    # Calculate difference
    total_distributed <- distributed_total + unmatched_total
    difference <- cams_total - total_distributed
    relative_diff <- ifelse(cams_total > 0, (difference / cams_total) * 100, 0)

    # Store results
    validation_results <- rbind(validation_results,
      data.frame(
        sector = sector_code,
        pollutant = poll_name,
        cams_total = cams_total,
        distributed = distributed_total,
        unmatched = unmatched_total,
        total_distributed = total_distributed,
        difference = difference,
        relative_diff_pct = relative_diff,
        stringsAsFactors = FALSE
      ))
  }
}
print(validation_results)
# =============================================================================
# 9. Add unmatched CAMS point source emissions to area sources
# =============================================================================

# Project area source rasters to domain and combine with unmatched emissions
# Approach: Keep area sources on coarse grid, aggregate unmatched emissions to coarse grid,
# combine both on coarse grid, then resample to domain grid. This ensures unmatched
# emissions are distributed uniformly within each coarse cell during resampling.

# Extract area source emissions BEFORE adding unmatched (for validation)
area_sources_before <- list()
area_rasters_coarse <- list()
augmented_area_sources <- list()
unmatched_to_area_summary <- data.frame()

for (sector_name in names(GNFR_codes)) {
  sector_code <- GNFR_codes[sector_name]

  if (is.null(cams_europe_asrc$rasters[[sector_code]])) {
    next
  }

  # Project and crop area sources to domain (keep on coarse grid for now)
  area_raster <- cams_europe_asrc$rasters[[sector_code]]
  area_raster_proj <- projectRaster(area_raster, crs = crs(domain))
  area_raster_proj <- crop(area_raster_proj, domain, snap = "out")

  # Extract emissions per pollutant from coarse grid (for validation)
  sector_totals <- data.frame()
  for (poll_idx in seq_len(nrow(pollutant_map))) {
    poll_name <- pollutant_map$cams[poll_idx]
    layer_name <- layer_name_map[[poll_name]]

    if (!is.null(layer_name) && layer_name %in% names(area_raster_proj)) {
      poll_values <- getValues(area_raster_proj[[layer_name]])
      poll_values[is.na(poll_values)] <- 0
      total_emission <- sum(poll_values, na.rm = TRUE)

      sector_totals <- rbind(sector_totals,
        data.frame(
          sector = sector_code,
          pollutant = poll_name,
          emission = total_emission,
          stringsAsFactors = FALSE
        ))
    }
  }

  if (nrow(sector_totals) > 0) {
    area_sources_before[[sector_code]] <- sector_totals
  }

  # Store coarse raster stack for combining with unmatched emissions
  area_rasters_coarse[[sector_name]] <- area_raster_proj
}

# Add unmatched emissions to area sources on coarse grid
for (sector_name in names(unmatched_emissions)) {
  sector_code <- GNFR_codes[sector_name]

  if (is.null(area_rasters_coarse[[sector_name]])) {
    next
  }

  # Work on the coarse raster stack
  area_raster_coarse <- area_rasters_coarse[[sector_name]]

  # For each pollutant with unmatched emissions, add to area sources on coarse grid
  for (poll_name in names(unmatched_emissions[[sector_name]])) {
    unmatched <- unmatched_emissions[[sector_name]][[poll_name]]

    if (nrow(unmatched) > 0) {
      layer_name <- layer_name_map[[poll_name]]
      if (is.null(layer_name) || !(layer_name %in% names(area_raster_coarse))) {
        next
      }

      # Convert unmatched points to sf and project to domain CRS
      unmatched_sf <- st_as_sf(unmatched,
                               coords = c("lon", "lat"),
                               crs = "+proj=longlat +datum=WGS84 +no_defs")
      unmatched_proj <- st_transform(unmatched_sf, crs = crs(domain))
      unmatched_sp <- as(unmatched_proj, "Spatial")

      # Rasterize unmatched emissions onto the COARSE grid (sum emissions per coarse cell)
      # This aggregates all unmatched points within each coarse CAMS cell
      unmatched_raster_coarse <- rasterize(unmatched_sp,
                                           area_raster_coarse[[layer_name]],
                                           field = "emission",
                                           fun = sum,
                                           background = 0,
                                           na.rm = TRUE)

      # Replace NA with 0
      unmatched_vals_coarse <- getValues(unmatched_raster_coarse)
      unmatched_vals_coarse[is.na(unmatched_vals_coarse)] <- 0
      unmatched_raster_coarse <- setValues(unmatched_raster_coarse, unmatched_vals_coarse)
      total_rasterized <- sum(unmatched_vals_coarse, na.rm = TRUE)

      # Add unmatched emissions to area sources on coarse grid
      area_vals_coarse <- getValues(area_raster_coarse[[layer_name]])
      area_vals_coarse[is.na(area_vals_coarse)] <- 0
      area_raster_coarse[[layer_name]] <- setValues(area_raster_coarse[[layer_name]],
                                                    area_vals_coarse + unmatched_vals_coarse)

      # Track summary
      original_unmatched_total <- sum(unmatched$emission, na.rm = TRUE)
      unmatched_to_area_summary <- rbind(
        unmatched_to_area_summary,
        data.frame(
          sector = sector_code,
          pollutant = poll_name,
          added_grid_cells = nrow(unmatched),
          original_unmatched = original_unmatched_total,
          rasterized_unmatched = total_rasterized,
          added_emissions = total_rasterized,
          rasterization_loss = original_unmatched_total - total_rasterized,
          stringsAsFactors = FALSE
        )
      )
    }
  }

  # Store updated coarse raster
  area_rasters_coarse[[sector_name]] <- area_raster_coarse
}

# Now resample combined coarse grids to domain grid
# This ensures unmatched emissions are distributed uniformly within each coarse cell
for (sector_name in names(area_rasters_coarse)) {
  area_raster_coarse <- area_rasters_coarse[[sector_name]]
  area_raster_domain <- resample(area_raster_coarse, domain, method = "ngb")
  augmented_area_sources[[sector_name]] <- area_raster_domain
}

# Save augmented area sources for use in area sources script
# The file will be loaded by 2_UrbEm_areasources_CAMS8.1_v2.R to replace GNFR_raster
# with area sources that include redistributed unmatched point source emissions
save(augmented_area_sources, file = paste0(output, "/", outputstring, "_corrected_areasources"))

# =============================================================================
# 10. Aggregate distributed point sources by location
# =============================================================================

# Combine all distributed points across sectors and pollutants
all_distributed <- data.frame()

for (sector_name in names(distributed_points)) {
  for (poll_name in names(distributed_points[[sector_name]])) {
    all_distributed <- rbind(all_distributed,
                            distributed_points[[sector_name]][[poll_name]])
  }
}

if (nrow(all_distributed) > 0) {
  # Create unique identifier for each point location
  all_distributed$point_id <- paste(all_distributed$xcor, all_distributed$ycor,
                                    all_distributed$sector, sep = "_")

  # Reshape to wide format: one row per point, columns for each pollutant
  all_distributed_wide <- dcast(all_distributed,
                               point_id + sector + xcor + ycor + lon + lat ~ pollutant,
                               value.var = "emission",
                               fun.aggregate = sum,
                               fill = 0)
} else {
  all_distributed_wide <- data.frame()
}

# =============================================================================
# 11. Format output for UECT (EPISODE-CityChem pre-processor)
# =============================================================================

if (nrow(all_distributed_wide) > 0) {
  # Map GNFR sectors to SNAP codes (Selected Nomenclature for Air Pollution)
  GNFR_to_SNAP <- c(
    "A" = 1,   # PublicPower
    "B" = 3,   # Industry
    "C" = 3,   # OtherStationaryComb
    "D" = 4,   # Fugitives
    "E" = 6,   # Solvents
    "F1" = 7,  # RoadTransport_Exhaust_Gasoline
    "F2" = 7,  # RoadTransport_Exhaust_Diesel
    "F3" = 7,  # RoadTransport_Exhaust_LPG_gas
    "F4" = 7,  # RoadTransport_NonExhaust
    "G" = 8,   # Shipping
    "H" = 8,   # Aviation
    "I" = 8,   # OffRoad
    "J" = 9,   # Waste
    "K" = 10,  # AgriLivestock
    "L" = 10   # AgriOther
  )

  # Add SNAP code
  all_distributed_wide$snap <- GNFR_to_SNAP[all_distributed_wide$sector]

  # Initialize pollutant columns if they don't exist
  for (poll_idx in seq_len(nrow(pollutant_map))) {
    cams_poll <- pollutant_map$cams[poll_idx]
    if (!cams_poll %in% names(all_distributed_wide)) {
      all_distributed_wide[[cams_poll]] <- 0
    }
  }

  # Create UECT output dataframe
  # Format: snap, xcor, ycor, Hi, Vi, Ti, radi, pollutants (NOx, NMVOC, CO, SO2, NH3, PM2.5, PM10, CH4, PN)
  # Pollutants in order: NOx, NMVOC, CO, SO2, NH3, PM2.5, PM10
  pollutant_order <- c("nox", "nmvoc", "co", "sox", "nh3", "pm2_5", "pm10")
  pollutant_uect_names <- c("NOx", "NMVOC", "CO", "SO2", "NH3", "PM2.5", "PM10")

  UrbEm_pointsources <- data.frame(
    snap = all_distributed_wide$snap,
    xcor = all_distributed_wide$xcor,
    ycor = all_distributed_wide$ycor,
    Hi = rep(-999, nrow(all_distributed_wide)),   # Stack height (not available)
    Vi = rep(-999, nrow(all_distributed_wide)),   # Exit velocity (not available)
    Ti = rep(-999, nrow(all_distributed_wide)),   # Exit temperature (not available)
    radi = rep(-999, nrow(all_distributed_wide)), # Stack radius (not available)
    stringsAsFactors = FALSE
  )

  # Add pollutant columns in correct order
  for (i in seq_along(pollutant_order)) {
    poll_cams <- pollutant_order[i]
    poll_uect <- pollutant_uect_names[i]

    if (poll_cams %in% names(all_distributed_wide)) {
      UrbEm_pointsources[[poll_uect]] <- all_distributed_wide[[poll_cams]]
    } else {
      UrbEm_pointsources[[poll_uect]] <- 0
    }
  }

  # Add CH4 if available
  if ("ch4" %in% names(all_distributed_wide)) {
    UrbEm_pointsources$CH4 <- all_distributed_wide$ch4
  }

  # Add PN dummy for CityChem 1.5
  UrbEm_pointsources$PN <- NA

  # Replace NA with -999 (UECT missing value indicator)
  UrbEm_pointsources[is.na(UrbEm_pointsources)] <- -999

  # Remove rows with zero coordinates (safety check)
  UrbEm_pointsources <- UrbEm_pointsources[!(UrbEm_pointsources$xcor == 0 &
                                             UrbEm_pointsources$ycor == 0),]

  # Write UECT input file
  setwd(output)
  write.csv(UrbEm_pointsources,
            paste0(outputstring, "_", nrow(UrbEm_pointsources), "_psrc.csv"),
            row.names = FALSE, quote = FALSE, na = "-999")
}

# =============================================================================
# 12. Clean environment
# =============================================================================

#rm(list = ls())
gc()
sum(UrbEm_pointsources$PM2.5)
#validation_results
unmatched_to_area_summary

####################
### END OF SCRIPT ###
####################

# Notes & caveats -------------------------------------------------------------
# • The augmented_area_sources object has the same structure as GNFR_raster: a simple
#   list keyed by full GNFR sector names (e.g., "A_PublicPower"). This allows it to
#   be directly assigned to GNFR_raster in the area sources script.
# • Emission distribution is proportional based on RI-URBANS emission weights within
#   each CAMS grid cell. If no RI-URBANS points exist in a cell, all CAMS emissions
#   for that cell are added to area sources to preserve total mass.
# • The unmatched emissions are rasterized onto the domain grid, which may result in
#   small losses if points fall outside the domain after projection. These losses
#   are tracked in the unmatched_to_area_summary output.

