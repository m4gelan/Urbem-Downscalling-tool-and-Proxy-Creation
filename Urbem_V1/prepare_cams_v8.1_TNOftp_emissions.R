# prepare_cams_emissions() — read & grid CAMS v8.1 emissions from NetCDF
# Annotated for clarity (2025-10). Original authors: M.O.P. Ramacher & A. Kakouri (2021)
#
# What this does
# ---------------
# • Opens a CAMS-REG-ANT v8.1 NetCDF file
# • Filters sources by *source type* (area/point/both) and *GNFR sector(s)*
# • Pulls emissions for chosen pollutants
# • Accumulates source emissions into the native CAMS lon/lat grid
# • Returns a list of raster *stacks* per sector (one layer per pollutant)
#
# Notes
# -----
# • Assumes the NetCDF contains variables:
#   - longitude_source, latitude_source     (per-source coords)
#   - emission_category_index               (per-source GNFR index 1..15)
#   - source_type_index                     (1 = area, 2 = point)
#   - longitude, latitude                   (grid axes for output gridding)
#   - pollutant variables: ch4, co, nh3, nmvoc, nox, pm10, pm2_5, sox
# • The function *accumulates* emissions into grid cells (sum over sources).
# • Raster latitude is flipped north-up for correct orientation.
# • `output_resolution` and `output_extent` are currently informational; the
#   function uses the NetCDF’s own grid axes.

prepare_cams_emissions <- function(
    nc_file_path,
    source_type = c("area", "point", "both"),  # filter which sources to include
    pollutants = c("ch4", "co", "nh3", "nmvoc", "nox", "pm10", "pm2_5", "sox"),
    sectors = NULL,   # NULL = all; or vector like c("A","B","F1",...) or GNFR names
    year = NULL,      # not used internally; kept for API symmetry
    output_resolution = 0.05,  # info-only; CAMS grid is read from file
    output_extent = NULL       # info-only; CAMS grid is read from file
) {
  # --- Dependencies ---------------------------------------------------------
  if (!require(ncdf4, quietly = TRUE)) {
    stop("Package 'ncdf4' is required but not installed")
  }
  
  # --- Open the NetCDF ------------------------------------------------------
  nc_file <- nc_open(nc_file_path)
  
  # --- Read indexing & metadata arrays --------------------------------------
  emis_cat_names     <- ncvar_get(nc_file, "emis_cat_name")          # sector labels
  emis_cat_indices   <- ncvar_get(nc_file, "emission_category_index") # sector index per source
  source_type_indices<- ncvar_get(nc_file, "source_type_index")       # 1=area, 2=point
  lons_source        <- ncvar_get(nc_file, "longitude_source")        # per-source longitudes
  lats_source        <- ncvar_get(nc_file, "latitude_source")         # per-source latitudes
  
  # GNFR index → code mapping (1..15 → A,B,C,D,E,F1,F2,F3,F4,G,H,I,J,K,L)
  gnfr_codes <- c("A","B","C","D","E","F1","F2","F3","F4","G","H","I","J","K","L")
  names(gnfr_codes) <- 1:15
  
  cat("Found", length(emis_cat_names), "emission categories in NetCDF file\n")
  for (i in seq_along(emis_cat_names)) {
    cat("  Sector", i, ":", gnfr_codes[i], "-", emis_cat_names[i], "\n")
  }
  
  # Pollutant key: file var name → pretty layer name
  pollutant_names <- c(CH4 = "CH4", CO = "CO", NH3 = "NH3", NMVOC = "NMVOC",
                       NOx = "NOx", PM10 = "PM10", `PM2.5` = "PM2.5", SO2 = "SO2")
  names(pollutant_names) <- c("ch4","co","nh3","nmvoc","nox","pm10","pm2_5","sox")
  
  # --- Validate & filter pollutant list -------------------------------------
  pollutants <- pollutants[pollutants %in% names(pollutant_names)]
  if (length(pollutants) == 0) stop("No valid pollutants specified")
  
  # --- Normalize/resolve requested sectors ----------------------------------
  # Accept inputs like c("A","B","F1") *or* full GNFR names like "A_PublicPower".
  if (is.null(sectors)) {
    sector_indices <- seq_along(emis_cat_names)  # all sectors present in file
    cat("Processing all", length(sector_indices), "sectors\n")
  } else {
    # If caller passed full GNFR names, strip to codes by splitting at underscore
    sec_codes <- sectors
    has_underscore <- grepl("_", sec_codes)
    if (any(has_underscore)) {
      sec_codes[has_underscore] <- sapply(strsplit(sec_codes[has_underscore], "_"), `[[`, 1)
    }
    # Map codes to 1..15 indices
    sector_indices <- unlist(lapply(sec_codes, function(code) which(gnfr_codes == code)))
    if (length(sector_indices) == 0) stop("No valid sectors specified")
    cat("Processing", length(sector_indices), "selected sectors:",
        paste(gnfr_codes[sector_indices], collapse = ", "), "\n")
  }
  
  # --- Build source-type mask -----------------------------------------------
  # Choose sources by type: area / point / both
  stype <- match.arg(source_type)
  if (stype == "area")      source_type_mask <- (source_type_indices == 1)
  else if (stype == "point")source_type_mask <- (source_type_indices == 2)
  else                       source_type_mask <- rep(TRUE, length(source_type_indices))
  
  # --- Sector mask (by emission category index) ------------------------------
  sector_mask <- emis_cat_indices %in% sector_indices
  
  # --- Combined mask & filtered vectors -------------------------------------
  combined_mask <- source_type_mask & sector_mask
  
  filtered_lons    <- lons_source[combined_mask]
  filtered_lats    <- lats_source[combined_mask]
  filtered_cat_idx <- emis_cat_indices[combined_mask]
  
  cat("Total sources in file:", length(lons_source), "\n")
  cat("Sources matching source type filter:", sum(source_type_mask), "\n")
  cat("Sources matching sector filter:",     sum(sector_mask),      "\n")
  cat("Sources matching both filters:",      sum(combined_mask),    "\n")
  
  # --- Read the CAMS grid axes (target raster grid) -------------------------
  cams_lon <- ncvar_get(nc_file, "longitude")
  cams_lat <- ncvar_get(nc_file, "latitude")
  cat("NetCDF grid dimensions:", length(cams_lon), "x", length(cams_lat), "\n")
  cat("Longitude range:", min(cams_lon), "to", max(cams_lon), "\n")
  cat("Latitude range:",  min(cams_lat), "to", max(cams_lat),  "\n")
  
  # --- Allocate emission grids per requested sector & pollutant -------------
  emission_grids <- vector("list", max(sector_indices))
  for (i in sector_indices) {
    emission_grids[[i]] <- lapply(pollutants, function(.)
      matrix(0, nrow = length(cams_lat), ncol = length(cams_lon)))
    names(emission_grids[[i]]) <- pollutants
  }
  cat("Created emission grids for sectors:", paste(gnfr_codes[sector_indices], collapse = ", "), "\n")
  
  # --- Extract emission values for filtered sources -------------------------
  all_emissions <- lapply(pollutants, function(p) ncvar_get(nc_file, p)[combined_mask])
  names(all_emissions) <- pollutants
  
  # --- Bin sources into grid cells & accumulate emissions -------------------
  cat("Processing", sum(combined_mask), "sources of type:", stype, "\n")
  cat("Grid dimensions:", length(cams_lon), "x", length(cams_lat), "\n")
  cat("Starting emission processing...\n")
  
  sources_per_sector <- setNames(rep(0L, length(sector_indices)), gnfr_codes[sector_indices])
  
  for (j in seq_along(filtered_lons)) {
    # Find nearest grid cell indices for this source location
    lon_idx <- which.min(abs(cams_lon - filtered_lons[j]))
    lat_idx <- which.min(abs(cams_lat - filtered_lats[j]))
    
    cat_idx <- filtered_cat_idx[j]  # 1..15 GNFR index
    
    # Count this source for its sector (only sectors we requested are present)
    sec_pos <- which(sector_indices == cat_idx)
    if (length(sec_pos) == 1) sources_per_sector[sec_pos] <- sources_per_sector[sec_pos] + 1L
    
    # Accumulate each pollutant into the grid cell
    for (p in pollutants) {
      emission_grids[[cat_idx]][[p]][lat_idx, lon_idx] <-
        emission_grids[[cat_idx]][[p]][lat_idx, lon_idx] + all_emissions[[p]][j]
    }
  }
  
  # --- Report per‑sector source counts --------------------------------------
  cat("Sources processed per sector:\n")
  for (i in seq_along(sources_per_sector)) {
    cat("  Sector", names(sources_per_sector)[i], ":", sources_per_sector[i], "sources\n")
  }
  
  # --- Close NetCDF ----------------------------------------------------------
  nc_close(nc_file)
  
  # --- Build raster stacks (one per sector) ---------------------------------
  cat("Creating raster stacks for", length(sector_indices), "sectors...\n")
  sector_rasters <- list()
  
  for (i in sector_indices) {
    cat("  Processing sector", i, "(", gnfr_codes[i], "-", emis_cat_names[i], ")...\n")
    
    if (length(emission_grids[[i]]) > 0) {
      sector_stack <- stack()
      for (p in pollutants) {
        grid_p <- emission_grids[[i]][[p]]
        if (nrow(grid_p) > 0 && ncol(grid_p) > 0) {
          cat("    Creating raster for", p, "...\n")
          r <- raster(grid_p,
                      xmn = min(cams_lon), xmx = max(cams_lon),
                      ymn = min(cams_lat), ymx = max(cams_lat),
                      crs = "+proj=longlat +datum=WGS84")
          # Ensure north-up orientation (CAMS latitude axis can be S→N or N→S)
          r <- flip(r, direction = "y")
          names(r) <- pollutant_names[p]
          sector_stack <- addLayer(sector_stack, r)
        }
      }
      sector_rasters[[gnfr_codes[i]]] <- sector_stack
      cat("    Sector", gnfr_codes[i], "completed with", nlayers(sector_stack), "pollutants\n")
    } else {
      cat("    Warning: No emission data for sector", gnfr_codes[i], "\n")
    }
  }
  
  # --- Return structured result ---------------------------------------------
  cat("Function completed successfully!\n")
  cat("Returning", length(sector_rasters), "sector rasters\n")
  
  list(
    rasters      = sector_rasters,          # named by GNFR code (e.g., "A","B",...)
    sectors      = sector_indices,          # numeric indices used (1..15 subset)
    sector_codes = gnfr_codes[sector_indices],
    pollutants   = pollutants,              # file var names used (e.g., "nox")
    source_type  = stype,
    grid_info    = list(
      lons = cams_lon,
      lats = cams_lat,
      resolution = output_resolution        # informational
    )
  )
}

# Example usage:
# area_sources <- prepare_cams_emissions(
#   nc_file_path = "path/to/file.nc",
#   source_type  = "area",
#   pollutants   = c("nox","sox"),
#   sectors      = c("A","B","C")   # or c("A_PublicPower","B_Industry",...)
# )
# point_sources <- prepare_cams_emissions(
#   nc_file_path = "path/to/file.nc",
#   source_type  = "point",
#   pollutants   = c("nox","sox"),
#   sectors      = c("A","B","C")
# )
