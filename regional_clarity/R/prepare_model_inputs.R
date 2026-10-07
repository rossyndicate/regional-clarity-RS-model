# Build model-ready inputs for the regional clarity SDD ensemble.
#
# The model needs, for each siteSR observation: the Landsat 7-harmonized
# bands, the atmospheric-correction flag, site characteristics (elevation,
# LakeCat catchment metrics, shore_flag), and antecedent gridMET weather.
# These functions assemble that table from AquaMatch siteSR data and the two
# web services, following 02b_pull_site_characteristics.Rmd and
# 02c_pull_weather_summaries.Rmd. 07_regional_application.Rmd uses them for
# the regional application, so this is the same code that produced its
# results.
#
# Quickest path, for sites and siteSR rows in AquaMatch's formats:
#
#   source("regional_clarity/R/prepare_model_inputs.R")
#   model_input <- prepare_model_inputs(sites, sitesr_rows)
#
# then predict and screen with regional_clarity/python/apply_model.py (or
# with the same steps in 07_regional_application.Rmd).
#
# `sites`: one row per siteSR_id, columns siteSR_id, WGS84_Latitude,
#   WGS84_Longitude, wb_nhd_id (NHDPlusV2 COMID), wb_nhd_source,
#   flag_wb, flag_optical_shoreline - as in AquaMatch's siteSR site table.
# `sitesr_rows`: siteSR observations that passed the scene QA in
#   01_filter_AquaMatch_Data.Rmd, columns siteSR_id, date, mission,
#   {red,green,blue,nir,swir1,swir2}_corr_7, surfacetemp_corr_7.

library(tidyverse)
library(sf)
library(elevatr)
library(StreamCatTools)
library(parallel)
library(climateR)
library(data.table)

# Landsat missions represented in training
TRAINING_MISSIONS <- c("LT04", "LT05", "LE07", "LC08", "LC09")

# identical to python/features.py; see CATCHMENT_WINSORIZE_CAP_SQKM there
CATCHMENT_WINSORIZE_CAP_SQKM <- 617.1

GRIDMET_RES <- 1 / 24
GRIDMET_LON0 <- -124.766666666667  # westernmost cell center
GRIDMET_LAT0 <- 49.4               # northernmost cell center
MAX_WINDOW_DAYS <- 30
WEATHER_WINDOWS <- c(1, 3, 7, 30)
GRIDMET_VARS <- c("pr", "tmmx", "tmmn", "srad")

.log <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), sprintf(...), "\n")


# Elevation and LakeCat catchment metrics per siteSR_id, plus shore_flag.
# Rows of `existing` (same columns, e.g. the training sites' values from 02b)
# are reused and only the remaining sites are fetched.
fetch_site_characteristics <- function(sites, existing = NULL) {
  stopifnot("some sites resolve to a non-NHDPlusv2 waterbody source - LakeCat join is not safe as-is" =
              all(sites$wb_nhd_source == "NHDPlusv2" | is.na(sites$wb_nhd_source)))

  existing <- if (is.null(existing)) {
    tibble(siteSR_id = character())
  } else {
    existing %>%
      filter(siteSR_id %in% sites$siteSR_id) %>%
      select(siteSR_id, elevation_m, catchment_area_sqkm, starts_with("pct_"))
  }
  to_fetch <- sites %>% filter(!siteSR_id %in% existing$siteSR_id)
  .log("site characteristics: %s reused, %s to fetch", nrow(existing), nrow(to_fetch))

  fetched <- NULL
  if (nrow(to_fetch) > 0) {
    pts_sf <- st_as_sf(to_fetch, coords = c("WGS84_Longitude", "WGS84_Latitude"), crs = 4326)
    elev <- get_elev_point(pts_sf, src = "aws", z = 9) %>%
      st_drop_geometry() %>%
      select(siteSR_id, elevation_m = elevation)

    lakecat_metrics <- paste(c(
      "pctimp2006", "pcturbhi2006", "pcturblo2006", "pcturbmd2006", "pcturbop2006",
      "pctconif2006", "pctdecid2006", "pctmxfst2006",
      "pcthbwet2006", "pctwdwet2006", "pctcrop2006"
    ), collapse = ",")
    comids <- unique(na.omit(as.character(to_fetch$wb_nhd_id)))
    chunks <- split(comids, ceiling(seq_along(comids) / 200))
    # a chunk where none of the comids are LakeCat-covered throws rather than
    # returning empty, so catch per chunk
    lakecat <- map(chunks, \(ch) {
      tryCatch(lc_get_data(comid = paste(ch, collapse = ","), metric = lakecat_metrics,
                           aoi = "catchment", showAreaSqKm = TRUE),
               error = function(e) { .log("LakeCat chunk failed: %s", conditionMessage(e)); tibble() })
    }) %>%
      list_rbind() %>%
      mutate(comid = as.character(comid),
             pct_forest_2006 = pctconif2006cat + pctdecid2006cat + pctmxfst2006cat,
             pct_wetland_2006 = pcthbwet2006cat + pctwdwet2006cat,
             pct_urban_2006 = pcturbhi2006cat + pcturblo2006cat + pcturbmd2006cat + pcturbop2006cat) %>%
      select(comid, catchment_area_sqkm = catareasqkm,
             pct_impervious_2006 = pctimp2006cat, pct_urban_2006, pct_forest_2006,
             pct_cropland_2006 = pctcrop2006cat, pct_wetland_2006)

    fetched <- to_fetch %>%
      mutate(comid = as.character(wb_nhd_id)) %>%
      select(siteSR_id, comid) %>%
      left_join(elev, by = "siteSR_id") %>%
      left_join(lakecat, by = "comid") %>%
      select(-comid)
  }

  # shore_flag follows the definition in 03_split_data.ipynb
  bind_rows(existing, fetched) %>%
    left_join(sites %>%
                transmute(siteSR_id,
                          shore_flag = as.numeric(flag_wb == 1 | flag_optical_shoreline == 1)),
              by = "siteSR_id")
}


# The gridMET cell (1/24 degree) each site falls in, with its center.
gridmet_cells <- function(sites) {
  sites %>%
    transmute(siteSR_id,
              col = round((WGS84_Longitude - GRIDMET_LON0) / GRIDMET_RES),
              row = round((GRIDMET_LAT0 - WGS84_Latitude) / GRIDMET_RES),
              cell_id = paste(col, row, sep = "_"),
              cell_lon = GRIDMET_LON0 + col * GRIDMET_RES,
              cell_lat = GRIDMET_LAT0 - row * GRIDMET_RES)
}


# The date range each cell needs: its observations plus the longest
# antecedent window (and a few days of margin).
gridmet_cell_ranges <- function(sitesr_rows, site_cells) {
  sitesr_rows %>%
    distinct(siteSR_id, date) %>%
    inner_join(site_cells, by = "siteSR_id") %>%
    summarise(min_date = min(date) - (MAX_WINDOW_DAYS + 5), max_date = max(date),
              .by = c(cell_id, cell_lon, cell_lat))
}


# Daily gridMET at each cell center, fetched in parallel (PSOCK workers). A
# failed cell is retried once. Returns list(weather = daily rows,
# failed = cells that still failed, with the error message).
fetch_gridmet <- function(cell_ranges, n_workers = 8) {
  fetch_one <- function(i, cell_ranges, vars) {
    row <- cell_ranges[i, ]
    fetch <- function() {
      pt <- sf::st_as_sf(data.frame(lon = row$cell_lon, lat = row$cell_lat),
                         coords = c("lon", "lat"), crs = 4326)
      res <- climateR::getGridMET(AOI = pt, varname = vars,
                                  startDate = as.character(row$min_date),
                                  endDate = as.character(row$max_date))
      res$cell_id <- row$cell_id
      res
    }
    tryCatch(fetch(), error = function(e) {
      Sys.sleep(5)
      tryCatch(fetch(), error = function(e2) data.frame(cell_id = row$cell_id, error = conditionMessage(e2)))
    })
  }

  t0 <- Sys.time()
  cl <- makeCluster(n_workers, type = "PSOCK")
  on.exit(stopCluster(cl))
  clusterEvalQ(cl, { suppressMessages(library(climateR)); suppressMessages(library(sf)) })
  results <- parLapply(cl, seq_len(nrow(cell_ranges)), fetch_one,
                       cell_ranges = cell_ranges, vars = GRIDMET_VARS)
  .log("gridMET fetch took %.1f min", as.numeric(difftime(Sys.time(), t0, units = "mins")))

  failed <- Filter(function(x) "error" %in% names(x), results)
  .log("succeeded: %s cells, failed: %s cells", length(results) - length(failed), length(failed))
  list(weather = bind_rows(Filter(function(x) !"error" %in% names(x), results)),
       failed = bind_rows(failed))
}


# 1-, 3-, 7- and 30-day antecedent summaries per cell and date: total
# precipitation, maximum daily high, mean temperature, minimum daily low, and
# mean solar radiation. Window w on `date` covers the w days strictly before
# the image date.
summarize_weather <- function(weather_raw) {
  weather <- as.data.table(weather_raw)[, .(cell_id, date = as.Date(date),
                                            precip_mm = pr,
                                            tmax_degC = tmmx - 273.15,
                                            tmin_degC = tmmn - 273.15,
                                            srad_Wm2 = srad)]
  weather[, tmean_degC := (tmax_degC + tmin_degC) / 2]

  # regular daily series per cell, so the rolling windows count calendar days
  full_days <- weather[, .(date = seq(min(date), max(date), by = "1 day")), by = cell_id]
  weather <- weather[full_days, on = .(cell_id, date)]
  setorder(weather, cell_id, date)

  for (w in WEATHER_WINDOWS) {
    weather[, paste0("precip_mm_prev", w) := frollsum(precip_mm, n = w), by = cell_id]
    weather[, paste0("tmax_degC_prev", w) := frollmax(tmax_degC, n = w), by = cell_id]
    weather[, paste0("tmean_degC_prev", w) := frollmean(tmean_degC, n = w), by = cell_id]
    weather[, paste0("tmin_degC_prev", w) := -frollmax(-tmin_degC, n = w), by = cell_id]
    weather[, paste0("srad_Wm2_prev", w) := frollmean(srad_Wm2, n = w), by = cell_id]
  }
  # shift so window w on `date` covers the w days strictly BEFORE date
  weather[, date := date + 1]
  as_tibble(weather[, c("cell_id", "date", grep("_prev", names(weather), value = TRUE)), with = FALSE])
}


# Spectral indices and site-feature coarsening: R ports of
# python/features.py (add_spectral_indices() and coarsen_site_features()).
add_features <- function(df) {
  df %>%
    mutate(BR = blue_corr7 / red_corr7,
           BG = blue_corr7 / green_corr7,
           NR = nir_corr7 / red_corr7,
           GR = green_corr7 / red_corr7,
           fai = nir_corr7 - (red_corr7 + (swir1_corr7 - red_corr7) * ((830 - 660) / (1650 - 660))),
           NDVI = (nir_corr7 - red_corr7) / (nir_corr7 + red_corr7),
           NDSSI = (blue_corr7 - nir_corr7) / (blue_corr7 + nir_corr7),
           NDWI = (green_corr7 - nir_corr7) / (green_corr7 + nir_corr7),
           MNDWI = (green_corr7 - swir1_corr7) / (green_corr7 + swir1_corr7)) %>%
    mutate(across(c(BR, BG, NR, GR, fai, NDVI, NDSSI, NDWI, MNDWI), ~ if_else(is.finite(.x), .x, NA_real_)),
           catchment_area_sqkm = if_else(catchment_area_sqkm > 0,
                                         10^round(log10(pmin(catchment_area_sqkm, CATCHMENT_WINSORIZE_CAP_SQKM)), 1),
                                         catchment_area_sqkm),
           across(c(pct_impervious_2006, pct_urban_2006, pct_forest_2006,
                    pct_cropland_2006, pct_wetland_2006), ~ round(.x, 0)))
}


# One model-input row per siteSR observation: harmonized bands, the LaSRC
# flag, site characteristics, and the antecedent weather for the site's
# gridMET cell on the image date, then spectral indices and coarsening.
# `keep` lists extra sitesr_rows columns to carry through.
assemble_model_inputs <- function(sitesr_rows, site_char, site_cells, weather_summaries, keep = character()) {
  sitesr_rows %>%
    select(siteSR_id, date, mission, all_of(keep),
           red_corr7 = red_corr_7, green_corr7 = green_corr_7, blue_corr7 = blue_corr_7,
           nir_corr7 = nir_corr_7, swir1_corr7 = swir1_corr_7, swir2_corr7 = swir2_corr_7,
           temp_corr7 = surfacetemp_corr_7) %>%
    mutate(date = as.Date(date),
           atm_corr_LaSRC = as.integer(mission %in% c("LC08", "LC09"))) %>%
    left_join(site_char, by = "siteSR_id") %>%
    left_join(site_cells %>% select(siteSR_id, cell_id), by = "siteSR_id") %>%
    left_join(as.data.frame(weather_summaries), by = c("cell_id", "date")) %>%
    add_features() %>%
    as_tibble()
}


# All of the above in one call. Weather for cells that fail to fetch is left
# missing, so those rows fail the completeness check downstream.
prepare_model_inputs <- function(sites, sitesr_rows, keep = character(), n_workers = 8) {
  sites <- sites %>% filter(siteSR_id %in% unique(sitesr_rows$siteSR_id))
  sitesr_rows <- sitesr_rows %>% mutate(date = as.Date(date))
  site_char <- fetch_site_characteristics(sites)
  site_cells <- gridmet_cells(sites)
  weather <- fetch_gridmet(gridmet_cell_ranges(sitesr_rows, site_cells), n_workers = n_workers)
  if (nrow(weather$failed) > 0) warning(nrow(weather$failed), " gridMET cells failed to fetch")
  assemble_model_inputs(sitesr_rows, site_char, site_cells, summarize_weather(weather$weather), keep = keep)
}
