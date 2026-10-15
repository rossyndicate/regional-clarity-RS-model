# Observation QA for AquaMatch siteSR data, as applied to the training data
# in 01_filter_AquaMatch_Data.Rmd ("Remote Sensing Filters" and "Filter
# reflectance outliers before matching"). 01 sources this file, so these are
# the definitions the training data were built with.
#
# Apply all three before prepare_model_inputs():
#
#   source("regional_clarity/R/siteSR_qa.R")
#   sites <- qa_sites(sites)
#   sitesr_rows <- sitesr_rows %>%
#     filter(siteSR_id %in% sites$siteSR_id) %>%
#     qa_siteSR_rows(scene_metadata) %>%
#     filter_band_outliers()
#
# The regional (HUC4) scoping in 01 is not repeated here; the training-region
# check (apply_model.py, step 07) flags observations outside it.

library(tidyverse)
library(parallel)

# harmonized (Landsat 7 reference) band columns the band-outlier filter fits
QA_BAND_COLS <- c("red_corr_7", "green_corr_7", "blue_corr_7",
                  "nir_corr_7", "swir1_corr_7", "swir2_corr_7", "surfacetemp_corr_7")


# Site geometry: keep sites whose buffer intersects exactly one waterbody.
# `sites` is AquaMatch's siteSR site table
# (siteSR_collated_WQP_NWIS_sites_with_NHD_info_*.csv).
qa_sites <- function(sites) {
  sites %>% filter(number_int_wb == 1)
}


# Pixel, temperature, and scene QA per observation. `scene_metadata` is the
# siteSR scene metadata (download_sceneMetadata()), joined on sat_id. Works on
# a data frame or a lazy arrow query.
qa_siteSR_rows <- function(sitesr_rows, scene_metadata) {
  clear_scenes <- scene_metadata %>% filter(CLOUD_COVER < 20) %>% pull(sat_id)
  sitesr_rows %>%
    filter(pCount_dswe1 / pCount_dswe_gt0 > 0.5,
           pCount_dswe1 > 8,
           # only rows with no clouds in the buffer area
           prop_clouds == 0,
           # no rows with median temp below freezing
           flag_temp_min == 0,
           # or above 40deg
           flag_temp_max == 0,
           # filter for scenes with < 20% cloud cover
           sat_id %in% clear_scenes)
}


# Per-site multiband RANSAC outlier filter over each site's full record.
# Sites with 10 or fewer observations can't be fit and pass unfiltered, as do
# sites where no iteration produced a usable fit (is_band_inlier NA), so run
# this on a site's whole siteSR history, not only the new scenes of interest.
# 01 fits one representative site per physical location and reuses its
# decision for co-located sites; fitting each siteSR_id separately, as here,
# is the same filter applied per site.
filter_band_outliers <- function(sitesr_rows, n_cores = max(1, detectCores() - 2), seed = 47) {
  set.seed(seed)
  sitesr_rows %>%
    arrange(siteSR_id, date) %>%
    group_by(siteSR_id) %>%
    group_split() %>%
    mclapply(function(site_df) {
      site_df %>%
        mutate(is_band_inlier = ransac_site_multiband(date, as.matrix(pick(all_of(QA_BAND_COLS)))))
    }, mc.cores = n_cores) %>%
    bind_rows() %>%
    filter(is_band_inlier | is.na(is_band_inlier)) %>%
    select(-is_band_inlier)
}


# Majority-vote across iterations (vote_frac >= cutoff), the same design as
# ransac_site() in 01_filter_AquaMatch_Data.Rmd - not winner-take-all (picking whichever single random
# iteration produced the most inliers), which this function used until it
# was redesigned to match: winner-take-all is far more sensitive to one
# lucky/unlucky random draw, and with only a 5-day match window and
# Landsat's sparse revisit, aggressively flagging scenes on that kind of
# noisy basis can zero out the *only* available candidate image for a field
# sample rather than just degrading match quality.
ransac_site_multiband <- function(date, band_mat,
                                  n_iter = 150,
                                  n_mad = 2.5,
                                  # per-band sensor-noise floor
                                  min_diff = c(red_corr_7 = 0.01, green_corr_7 = 0.01,
                                               blue_corr_7 = 0.01, nir_corr_7 = 0.01,
                                               swir1_corr_7 = 0.01, swir2_corr_7 = 0.01,
                                               surfacetemp_corr_7 = 1),
                                  min_sample = 10,
                                  vote_frac_cutoff = 0.5) {
  n <- nrow(band_mat)
  nb <- ncol(band_mat)
  if (n < min_sample + 1) return(rep(NA, n))  # too few points to run ransac

  # match by name rather than assuming column order lines up with min_diff
  min_diff_b <- min_diff[colnames(band_mat)]

  t <- as.numeric(date)

  # fixed per-band residual scale from a single all-data fit (MAD is robust
  # up to ~50% contamination, well above what we expect here) - reused for
  # every candidate subsample below instead of recomputed per iteration
  band_scale <- vapply(seq_len(nb), function(b) {
    y <- band_mat[, b]
    cf <- ols_fit(t, y)
    pred <- cf["intercept"] + cf["slope"] * t
    resid <- y - pred
    s <- mad(resid, na.rm = TRUE)
    if (is.na(s) || s == 0) s <- sd(resid, na.rm = TRUE)
    s
  }, numeric(1))

  # tally, per point, how many valid iterations called it an inlier
  inlier_votes <- rep(0L, n)
  n_valid_iter <- 0L

  for (i in 1:n_iter) {
    idx <- sample(n, min_sample)
    if (length(unique(t[idx])) < 2) next  # avoid degenerate fit (same date)

    # fit each band vs time on the *same* subsample, standardize each band's
    # residuals by its fixed scale, then combine - keeps one band's scale
    # from dominating and keeps the threshold comparable across candidate
    # lines of very different quality
    z <- matrix(NA_real_, nrow = n, ncol = nb)
    for (b in seq_len(nb)) {
      y <- band_mat[, b]
      if (length(unique(y[idx])) < 2 || is.na(band_scale[b]) || band_scale[b] == 0) next

      cf <- ols_fit(t[idx], y[idx])
      if (anyNA(cf)) next
      pred <- cf["intercept"] + cf["slope"] * t
      resid <- y - pred
      z[, b] <- abs(resid) / band_scale[b]
      # a raw difference this small is within sensor noise for this band -
      # never let it count as outlier evidence, regardless of how tight
      # band_scale is
      z[abs(resid) < min_diff_b[b], b] <- 0
    }

    combined_z <- rowMeans(z, na.rm = TRUE)
    inliers <- !is.na(combined_z) & combined_z < n_mad

    inlier_votes <- inlier_votes + as.integer(inliers)
    n_valid_iter <- n_valid_iter + 1L
  }

  # no iteration ever produced a usable fit - we haven't determined
  # anything about this site, so say so rather than leaving every point
  # flagged by the rep(0L, n) initialization
  if (n_valid_iter == 0) return(rep(NA, n))

  # a point is an inlier if at least vote_frac_cutoff of the valid
  # iterations agreed
  best_inliers <- (inlier_votes / n_valid_iter) >= vote_frac_cutoff

  # a flagged point can't be implausible for this site if every one of its
  # band values falls within the range other accepted (inlier) points
  # already span for that band - mirrors the same rescue in ransac_site() (01)
  if (any(best_inliers)) {
    in_range <- vapply(seq_len(nb), function(b) {
      y <- band_mat[, b]
      accepted <- y[best_inliers]
      if (length(accepted) == 0 || all(is.na(accepted))) return(rep(TRUE, n))
      rng <- range(accepted, na.rm = TRUE)
      y >= rng[1] & y <= rng[2]
    }, logical(n))
    best_inliers <- best_inliers | apply(in_range, 1, all)
  }

  best_inliers
}

# define closed-form OLS slope/intercept for y ~ t
ols_fit <- function(t, y) {
  ok <- is.finite(t) & is.finite(y)
  if (sum(ok) < 2 || length(unique(t[ok])) < 2) {
    return(c(intercept = NA_real_, slope = NA_real_))
  }
  tt <- t[ok]; yy <- y[ok]
  tm <- mean(tt); ym <- mean(yy)
  slope <- sum((tt - tm) * (yy - ym)) / sum((tt - tm)^2)
  c(intercept = ym - slope * tm, slope = slope)
}
