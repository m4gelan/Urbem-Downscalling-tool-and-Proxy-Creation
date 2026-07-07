fetch_osm_roads <- function(bbox, street_types) {
  servers <- c(
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter"
  )
  last_err <- NULL
  for (url in servers) {
    err <- NULL
    lines <- tryCatch({
      osmdata::set_overpass_url(url)
      q <- add_osm_feature(opq(bbox = bbox, timeout = 180), "highway", street_types)
      result <- osmdata_sf(q, quiet = TRUE)
      if (is.null(result$osm_lines) || nrow(result$osm_lines) == 0) {
        stop("no road lines returned")
      }
      result$osm_lines[, c("highway"), drop = FALSE]
    }, error = function(e) {
      err <<- conditionMessage(e)
      NULL
    })
    if (!is.null(lines)) {
      print(paste("OSM roads downloaded from", url))
      return(lines)
    }
    last_err <- err
    print(paste("Overpass failed (", url, "):", err))
  }
  stop("OSM download failed on all Overpass servers. Last error: ", last_err)
}

area_to_osm_lines <- function(domain, area_emissions, osm_cache = NULL)
{
  ### prepare spatial dataframe objects for processing of line sources
  # our domain
  domain_sf <- rasterToPolygons(domain)
  domain_sf <- st_as_sf(domain_sf)
  #domain_sf <- st_transform(domain_sf, crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

  # our emissions as sf in right projection to read osm data
  emis <- rasterToPolygons(area_emissions)
  emis <- st_as_sf(emis)
  emis <- st_transform(emis, crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

  street_types <- c("motorway", "trunk", "primary", "secondary", "tertiary")

  if (!is.null(osm_cache) && file.exists(osm_cache)) {
    print(paste("loading cached OSM roads:", osm_cache))
    osm <- st_read(osm_cache, quiet = TRUE)
  } else {
    print("downloading OSM data for specified domain")
    osm <- fetch_osm_roads(st_bbox(emis), street_types)
    if (!is.null(osm_cache)) {
      cache_dir <- dirname(osm_cache)
      if (!dir.exists(cache_dir)) dir.create(cache_dir, recursive = TRUE)
      st_write(osm, osm_cache, quiet = TRUE, delete_dsn = TRUE)
      print(paste("cached OSM roads:", osm_cache))
    }
  }

  ### transform back to UTM projection and crop it to the correct domain exten
  osm <- st_transform(osm, crs = crs(domain))
  
  ### distribution of all emissions to selected road types including 
  ### (a) road elements lengths
  ### (b) weighting by road type and for each cell separately
  options(warn=-1)
  #remove(out)
  print("distributing area emissions: each grid cell value is attributed to road elements that fall into a grid cell")
  
  # initiate progress bar
  progress = txtProgressBar(min = 0, max = length(domain_sf$geometry), initial = 0, style = 3) 
  
  for(i in 1:length(domain_sf$geometry))
  {
    
    #algorithm based on VEIN package's "emis_dist" function by S. Ibara
    #modified by M. Ramacher
    
    ### gather all roads, which are in cell i from emissions raster
    osm_cell <- st_intersection(osm,domain_sf$geometry[i])
    
    ### what road types are in the gathered roads
    st <- unique(osm_cell$highway)
    
    ### create road-type-weighting for cell i based on road types in cell i
    
    if(length(st)>0)
    {
      ### how do we weight each road type?
      ## motorway & motorway link     = 10
      ## trunk & trunk linkl          = 5
      ## primary & primary link       = 2
      ## secondary & secondary link   = 2
      ## tertiary & tertiary link     = 1
      
      ### initalize weights with 0
      road_weights <- c(0,0,0,0,0)
      
      ### each road type gets a weighting if the road type exists in cell i
      if(any(grepl("motorway", st)))
      {
        road_weights[1] <- 10
      }
      if(any(grepl("trunk", st)))
      {
        road_weights[2] <- 5
      }
      if(any(grepl("primary", st)))
      {
        road_weights[3] <- 2
      }
      if(any(grepl("secondary", st)))
      {
        road_weights[4] <- 2
      }
      if(any(grepl("tertiary", st)))
      {
        road_weights[5] <- 1
      }
      
      ### normalize weighting factors individually for cell i
      normalized_road_weight <- road_weights/sum(road_weights)
      
      ## length of streets (line elements) in cell i
      osm_cell$lkm1 <- as.numeric(sf::st_length(osm_cell))
      
      ### prepare table for distributing emissions of cell i:
      osm_cell <- osm_cell[, c("highway", "lkm1")]
      
      ### gather emissions for cell i from emissions raster
      NOx <- emis$NOx[i]
      NMVOC <- emis$NMVOC[i]
      CO <- emis$CO[i]
      SO2 <- emis$SO2[i]
      NH3 <- emis$NH3[i]
      PM25 <- emis$PM2.5[i]
      PM10 <- emis$PM10[i]
      
      ### select motorways and distribute emissions of cell i to motorways using normalized road weight AND street length
      ### lenght of motorway / sum of motorway length * weighting factor * total emissions
      osm_cell_m <- osm_cell[grep("motorway",osm_cell$highway),]
      osm_cell_m$NOx <- osm_cell_m$lkm1/sum(osm_cell_m$lkm1) * normalized_road_weight[1] * NOx
      osm_cell_m$NMVOC <- osm_cell_m$lkm1/sum(osm_cell_m$lkm1) * normalized_road_weight[1] * NMVOC
      osm_cell_m$CO <- osm_cell_m$lkm1/sum(osm_cell_m$lkm1) * normalized_road_weight[1] * CO
      osm_cell_m$SO2 <- osm_cell_m$lkm1/sum(osm_cell_m$lkm1) * normalized_road_weight[1] * SO2
      osm_cell_m$NH3 <- osm_cell_m$lkm1/sum(osm_cell_m$lkm1) * normalized_road_weight[1] * NH3
      osm_cell_m$PM25 <- osm_cell_m$lkm1/sum(osm_cell_m$lkm1) * normalized_road_weight[1] * PM25
      osm_cell_m$PM10 <- osm_cell_m$lkm1/sum(osm_cell_m$lkm1) * normalized_road_weight[1] * PM10
      
      ### select trunks
      osm_cell_t <- osm_cell[grep("trunk",osm_cell$highway), ]
      osm_cell_t$NOx <- osm_cell_t$lkm1/sum(osm_cell_t$lkm1) * normalized_road_weight[2] * NOx
      osm_cell_t$NMVOC <- osm_cell_t$lkm1/sum(osm_cell_t$lkm1) * normalized_road_weight[2] * NMVOC
      osm_cell_t$CO <- osm_cell_t$lkm1/sum(osm_cell_t$lkm1) * normalized_road_weight[2] * CO
      osm_cell_t$SO2 <- osm_cell_t$lkm1/sum(osm_cell_t$lkm1) * normalized_road_weight[2] * SO2
      osm_cell_t$NH3 <- osm_cell_t$lkm1/sum(osm_cell_t$lkm1) * normalized_road_weight[2] * NH3
      osm_cell_t$PM25 <- osm_cell_t$lkm1/sum(osm_cell_t$lkm1) * normalized_road_weight[2] * PM25
      osm_cell_t$PM10 <- osm_cell_t$lkm1/sum(osm_cell_t$lkm1) * normalized_road_weight[2] * PM10
      
      ### primarys
      osm_cell_p <- osm_cell[grep("primary",osm_cell$highway), ]
      osm_cell_p$NOx <- osm_cell_p$lkm1/sum(osm_cell_p$lkm1) * normalized_road_weight[3] * NOx
      osm_cell_p$NMVOC <- osm_cell_p$lkm1/sum(osm_cell_p$lkm1) * normalized_road_weight[3] * NMVOC
      osm_cell_p$CO <- osm_cell_p$lkm1/sum(osm_cell_p$lkm1) * normalized_road_weight[3] * CO
      osm_cell_p$SO2 <- osm_cell_p$lkm1/sum(osm_cell_p$lkm1) * normalized_road_weight[3] * SO2
      osm_cell_p$NH3 <- osm_cell_p$lkm1/sum(osm_cell_p$lkm1) * normalized_road_weight[3] * NH3
      osm_cell_p$PM25 <- osm_cell_p$lkm1/sum(osm_cell_p$lkm1) * normalized_road_weight[3] * PM25
      osm_cell_p$PM10 <- osm_cell_p$lkm1/sum(osm_cell_p$lkm1) * normalized_road_weight[3] * PM10
      
      ### secondaries
      osm_cell_s <- osm_cell[grep("secondary",osm_cell$highway), ]
      osm_cell_s$NOx <- osm_cell_s$lkm1/sum(osm_cell_s$lkm1) * normalized_road_weight[4] * NOx
      osm_cell_s$NMVOC <- osm_cell_s$lkm1/sum(osm_cell_s$lkm1) * normalized_road_weight[4] * NMVOC
      osm_cell_s$CO <- osm_cell_s$lkm1/sum(osm_cell_s$lkm1) * normalized_road_weight[4] * CO
      osm_cell_s$SO2 <- osm_cell_s$lkm1/sum(osm_cell_s$lkm1) * normalized_road_weight[4] * SO2
      osm_cell_s$NH3 <- osm_cell_s$lkm1/sum(osm_cell_s$lkm1) * normalized_road_weight[4] * NH3
      osm_cell_s$PM25 <- osm_cell_s$lkm1/sum(osm_cell_s$lkm1) * normalized_road_weight[4] * PM25
      osm_cell_s$PM10 <- osm_cell_s$lkm1/sum(osm_cell_s$lkm1) * normalized_road_weight[4] * PM10
      
      ### tertiaries
      osm_cell_te <- osm_cell[grep("tertiary",osm_cell$highway), ]
      osm_cell_te$NOx <- osm_cell_te$lkm1/sum(osm_cell_te$lkm1) * normalized_road_weight[5] * NOx
      osm_cell_te$NMVOC <- osm_cell_te$lkm1/sum(osm_cell_te$lkm1) * normalized_road_weight[5] * NMVOC
      osm_cell_te$CO <- osm_cell_te$lkm1/sum(osm_cell_te$lkm1) * normalized_road_weight[5] * CO
      osm_cell_te$SO2 <- osm_cell_te$lkm1/sum(osm_cell_te$lkm1) * normalized_road_weight[5] * SO2
      osm_cell_te$NH3 <- osm_cell_te$lkm1/sum(osm_cell_te$lkm1) * normalized_road_weight[5] * NH3
      osm_cell_te$PM25 <- osm_cell_te$lkm1/sum(osm_cell_te$lkm1) * normalized_road_weight[5] * PM25
      osm_cell_te$PM10 <- osm_cell_te$lkm1/sum(osm_cell_te$lkm1) * normalized_road_weight[5] * PM10
      
      ### unite all weigthed road type emissions
      osm_cell_all <- rbind(osm_cell_m,osm_cell_t, osm_cell_p, osm_cell_s, osm_cell_te)
      head(osm_cell_all)
      
      ### dismiss length of road types column
      osm_cell_all <- osm_cell_all[, c("NOx", "NMVOC", "CO", "SO2", "NH3", "PM25", "PM10", "highway")]
      
      ### give correct names
      names(osm_cell_all) <- c("NOx", "NMVOC", "CO", "SO2", "NH3", "PM25", "PM10", "roadtype", "geometry")
      
      ### plot each iteration
      #plot(osm_cell_all["NOx"])
      #print(sum(osm_cell_all$NOx))
      
      if(!exists("out"))
      {
        out <- osm_cell_all
      } else {
        out <- rbind(out, osm_cell_all)
      }
    }
    setTxtProgressBar(progress,i)
  }
  options(warn=0)
  return(out)
}