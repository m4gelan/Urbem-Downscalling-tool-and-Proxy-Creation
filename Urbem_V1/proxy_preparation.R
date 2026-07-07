# =============================================================================
# PROXY PREPARATION FUNCTIONS
# =============================================================================
# Functions for normalizing and applying proxy datasets to downscale coarse
# CAMS emission grids to fine-resolution domain grids.
#
# Author: M.O.P. Ramacher, Hereon
# Date: 2025
# =============================================================================


# -----------------------------------------------------------------------------
# proxy_cwd: Cell-Wise Distribution of Proxy Weights
# -----------------------------------------------------------------------------
# Creates normalized proxy weights for downscaling coarse CAMS emissions to
# fine-resolution domain grids. The function processes each coarse CAMS cell
# individually, normalizing proxy values within each cell so that weights sum
# to 1.0 per coarse cell.
#
# Algorithm:
#   1. Create unique ID raster from coarse CAMS cells to identify each cell
#      (avoids issues when many cells have identical emission values, e.g., 0)
#   2. Resample ID raster to domain grid using nearest neighbor
#   3. For each unique coarse cell:
#      a. Identify all fine cells belonging to this coarse cell
#      b. Crop proxy to this coarse cell extent
#      c. Resample proxy to match fine cell grid exactly
#      d. Normalize proxy values so they sum to 1.0 within the coarse cell
#      e. Handle NA values by redistributing mass uniformly
#      f. Apply boundary correction for incomplete cells (<80% coverage)
#   4. Merge all normalized tiles into a single raster
#
# Parameters:
#   cams_origin: RasterLayer
#     Coarse-resolution CAMS raster (single layer) in its original projection.
#     Used to define the coarse cell structure. All sectors use the same grid.
#
#   domain: RasterLayer or RasterStack
#     Fine-resolution target domain grid. Defines the output resolution and
#     extent. Must be in the same CRS as the final output.
#
#   proxy: RasterLayer
#     Proxy dataset (e.g., population, industry) on the domain grid.
#     Should already be projected and cropped to domain extent.
#
#   sparse_threshold: numeric (default: 0.05)
#     If fewer than this fraction of cells have non-zero proxy values within
#     a coarse cell, apply hybrid distribution (proxy + uniform) to avoid
#     unrealistic hotspots. Value between 0 and 1.
#
#   max_weight_per_cell: numeric (default: 0.5)
#     Maximum fraction of a coarse cell's emissions that can be allocated to
#     a single fine cell. Excess is redistributed uniformly. Prevents extreme
#     concentration in single cells.
#
# Returns:
#   RasterLayer
#     Normalized proxy weights on the domain grid. Within each coarse CAMS cell,
#     the weights sum to 1.0 (or less for boundary cells with <80% coverage).
#     The sum across all cells equals approximately the number of coarse cells.
#
# Notes:
#   - NA values in proxy are replaced with 0 BEFORE processing
#   - Uses ID-based approach to handle cases where many coarse cells have
#     identical values (common for point sources with many zero-emission cells)
#   - Boundary correction scales weights for incomplete cells (<80% coverage)
#     to prevent over-allocation at domain edges
#   - Uniform distributions use max_cell (expected cell count) to ensure
#     consistent values across coarse cells with different actual cell counts
#   - Sparse proxy coverage (< sparse_threshold) triggers hybrid distribution
#     to prevent unrealistic hotspots from single proxy cells
#   - Maximum weight cap prevents excessive concentration in single cells
# -----------------------------------------------------------------------------
proxy_cwd <- function(cams_origin, domain, proxy,
                       sparse_threshold = 0.05,  # If <5% of cells have proxy, use hybrid
                       max_weight_per_cell = 0.5)  # No single cell gets >50% of coarse cell
{
  #### plot proxy
  plot(proxy)
  print(paste0("processing: ", names(proxy)))

  # Replace NA values with 0 BEFORE processing
  proxy[is.na(proxy)] <- 0

  ### Create ID raster from coarse CAMS cells to uniquely identify each coarse cell
  ## This avoids problems when many coarse cells have the same value (e.g., 0 for point sources)
  cams_origin_ids <- cams_origin
  cams_origin_ids[] <- 1:ncell(cams_origin)  # Unique ID per coarse cell

  ## Resample ID raster to domain grid (same method as emissions: nearest neighbor)
  cams_domain_ids <- resample(cams_origin_ids, domain, method = "ngb")

  ### Gather unique cell IDs to identify "bigger" cells from original CAMS raster
  ## Each unique ID represents one coarse CAMS cell
  conc_val <- unique(values(cams_domain_ids))
  conc_val <- conc_val[!is.na(conc_val)]  # Remove NA values

  ## Calculate amount of "smaller" cells in the resampled CAMS data,
  ## that fit into the "bigger" cells of the original CAMS raster
  ## based on resolutions of the different raster
  max_cell <- prod(res(cams_origin)/res(domain))

  ## Initialize lists
  ids <- list()
  proxy_norm_tiles <- list()

  ## For every unique coarse cell ID ...
  for(c in 1:length(conc_val))
  {
    ## Loop to collect cell coordinates of cells belonging to this coarse cell
    n <- 1
    i_id <- array()
    j_id <- array()

    for(i in 1:ncol(cams_domain_ids))
    {
      for(j in 1:nrow(cams_domain_ids))
      {
        # Check if this cell belongs to the current coarse cell (by ID)
        if(any(cams_domain_ids[j,i]==conc_val[c]))
        {
          i_id <- c(i_id,j)
          j_id <- c(j_id,i)
          n=n+1
        } else
        {
          n=n+1
        }
      }
    }
    ## Gather blocks of same cell IDs in one raster
    ## by cropping it with the min and max cell coordinates from the domain
    ## and store it in a list of rasters
    ids[[c]] <- crop(cams_domain_ids,extent(cams_domain_ids,
                                        min(i_id, na.rm = T),
                                        max(i_id, na.rm = T),
                                        min(j_id, na.rm = T),
                                        max(j_id, na.rm = T)))

    #### Crop proxy to this coarse cell extent
    proxy_cropped <- crop(proxy, ids[[c]])

    # Resample to ensure exact match with ids[[c]] (same extent, resolution, and cells)
    # This ensures proxy_norm_tiles has exactly the same structure as ids[[c]]
    if (ncell(proxy_cropped) > 0 && !is.null(values(proxy_cropped))) {
      # Resample to match ids[[c]] exactly
      proxy_resampled <- resample(proxy_cropped, ids[[c]], method = "ngb")
      proxy_sum <- sum(values(proxy_resampled), na.rm = T)

      # Count cells with non-zero proxy values for sparsity detection
      proxy_vals_raw <- values(proxy_resampled)
      n_nonzero <- sum(proxy_vals_raw > 0, na.rm = T)
      n_total <- length(proxy_vals_raw[!is.na(proxy_vals_raw)])
      coverage_fraction <- ifelse(n_total > 0, n_nonzero / n_total, 0)

      #### Normalize proxy values within this coarse cell
      if (proxy_sum > 0)
      {
        # Check if all proxy values are the same (uniform proxy, e.g., all = 1)
        # This ensures consistent values across coarse cells with different actual cell counts
        unique_nonzero_vals <- unique(proxy_vals_raw[proxy_vals_raw > 0])
        is_uniform_proxy <- length(unique_nonzero_vals) == 1

        if (is_uniform_proxy) {
          # Uniform proxy: use uniform distribution with max_cell for consistency
          proxy_norm_tiles[[c]] <- ids[[c]]
          proxy_norm_tiles[[c]][] <- 1/max_cell
        } else {
          # Non-uniform proxy: normalize and handle sparse coverage
          proxy_norm_tiles[[c]] <- proxy_resampled
          proxy_norm_tiles[[c]] <- proxy_norm_tiles[[c]] / proxy_sum
          proxy_vals <- values(proxy_norm_tiles[[c]])

          # Handle very sparse cases (1-3 cells with proxy)
          if (n_nonzero > 0 && n_nonzero <= 3) {
            proxy_weight <- min(n_nonzero / 10, 0.3)  # 10-30% proxy weight
            uniform_dist <- rep(1/max_cell, n_total)
            proxy_vals <- (1 - proxy_weight) * uniform_dist + proxy_weight * proxy_vals
            proxy_vals <- proxy_vals / sum(proxy_vals, na.rm = T)
          }
          # Handle sparse coverage (but not extremely sparse)
          else if (coverage_fraction < sparse_threshold && coverage_fraction > 0) {
            uniform_weight <- min(1 - (coverage_fraction / sparse_threshold), 0.8)
            uniform_dist <- rep(1/max_cell, n_total)
            proxy_vals <- (1 - uniform_weight) * proxy_vals + uniform_weight * uniform_dist
            proxy_vals <- proxy_vals / sum(proxy_vals, na.rm = T)
          }

          # Cap maximum weight per cell to avoid extreme hotspots
          if (n_nonzero > 0 && n_nonzero <= 3) {
            dynamic_max_weight <- max(0.1, max_weight_per_cell * (n_nonzero / 3))
          } else if (coverage_fraction < sparse_threshold && coverage_fraction > 0) {
            dynamic_max_weight <- max_weight_per_cell * (1 - (sparse_threshold - coverage_fraction))
          } else {
            dynamic_max_weight <- max_weight_per_cell
          }

          max_weight <- max(proxy_vals, na.rm = T)
          if (max_weight > dynamic_max_weight) {
            excess_indices <- which(proxy_vals > dynamic_max_weight)
            excess_mass <- sum(proxy_vals[excess_indices] - dynamic_max_weight)
            proxy_vals[excess_indices] <- dynamic_max_weight
            other_indices <- which(proxy_vals <= dynamic_max_weight)
            if (length(other_indices) > 0) {
              proxy_vals[other_indices] <- proxy_vals[other_indices] + excess_mass / length(other_indices)
            }
            proxy_vals <- proxy_vals / sum(proxy_vals, na.rm = T)
          }

          proxy_norm_tiles[[c]][] <- proxy_vals
        }
      } else {
        # No proxy values: use uniform distribution with max_cell for consistency
        proxy_norm_tiles[[c]] <- ids[[c]]
        proxy_norm_tiles[[c]][] <- 1/max_cell
      }
    } else {
      # Proxy doesn't overlap: use uniform distribution with max_cell for consistency
      proxy_norm_tiles[[c]] <- ids[[c]]
      proxy_norm_tiles[[c]][] <- 1/max_cell
    }

    ### Correct grid cell values with less than 80% coverage of the original CAMS coarse resolution
    ### by applying the proportion of resolutions to all grid cells, that are affected
    ### otherwise we have values that are too high (especially at the boundaries)
    ### Only apply correction if coverage is significantly less than 100% (< 0.8)
    coverage_ratio <- ncell(ids[[c]])/max_cell
    if (coverage_ratio < 0.8)
    {
      proxy_norm_tiles[[c]] <- proxy_norm_tiles[[c]] * coverage_ratio
    }
  }

  ### Merge all normalized proxy tiles into one raster
  proxy_norm <- proxy_norm_tiles[[1]]
  for(i in 2:length(proxy_norm_tiles))
  {
    proxy_norm <- merge(proxy_norm, proxy_norm_tiles[[i]])
  }

  #### plot proxy_norm
  plot(proxy_norm)

  return(proxy_norm)
}


# -----------------------------------------------------------------------------
# proxy_distribution: Apply Proxy Weights to Emissions
# -----------------------------------------------------------------------------
# Applies normalized proxy weights to emission rasters to downscale emissions
# from coarse to fine resolution using spatial proxies.
#
# Two methods are supported:
#   1. "top_down_proxy": Total emissions are preserved exactly. The sum of
#      all fine cells equals the sum of the coarse cells.
#   2. "coarse_cells_proxy": Emissions are distributed proportionally within
#      each coarse cell according to proxy weights. Mass conservation is
#      maintained per coarse cell.
#
# Parameters:
#   emissions: RasterStack or RasterBrick
#     Emission rasters (one layer per pollutant) to be downscaled.
#     Should be on the coarse CAMS grid or already resampled to domain grid.
#
#   proxy: RasterLayer
#     Normalized proxy weights (typically from proxy_cwd function).
#     Should be on the same grid as the target domain.
#
#   proxy_method: character
#     Method for applying proxy: "top_down_proxy" or "coarse_cells_proxy"
#
# Returns:
#   RasterBrick
#     Downscaled emission rasters with one layer per pollutant.
#     Layer names are preserved from the input emissions.
#
# Notes:
#   - For "top_down_proxy", the total sum across all cells is preserved
#   - For "coarse_cells_proxy", mass is conserved within each coarse cell
#   - Both methods maintain spatial patterns according to proxy distribution
# -----------------------------------------------------------------------------
proxy_distribution <- function(emissions,proxy,proxy_method)
{
  plot(emissions)
  temp <- list()
  for (j in 1:nlayers(emissions))
  {
    if (proxy_method == "top_down_proxy")
    {
      # Preserve total emissions: multiply total sum by normalized proxy
      temp[[j]] <- sum(getValues(emissions[[j]]), na.rm = T)*proxy
    }
    if (proxy_method == "coarse_cells_proxy")
    {
      # Distribute emissions proportionally within each coarse cell
      temp[[j]] <- emissions[[j]]*proxy
    }
    names(temp[[j]]) <- names(emissions[[j]])
  }

  ### Create rasterbrick out of different pollutant rasterlayers
  emissions <- brick(temp)
  plot(emissions)

  return(emissions)
}
