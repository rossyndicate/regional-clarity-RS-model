"""Spectral index feature engineering.

Optical candidates are curated to raw bands plus ratios/indices with
specific literature precedent or clear physical explainability, rather
than an exhaustive sweep of every possible band combination:

  - BR (blue/red): Kloiber et al.'s canonical Landsat Secchi-depth ratio,
    the basis of the statewide Minnesota/Wisconsin clarity mapping
    programs.
  - BG (blue/green): classic ocean-color chlorophyll-a ratio.
  - NR (nir/red): established chlorophyll-a ratio for turbid inland
    waters.
  - GR (green/red): same chlorophyll-absorption-contrast logic as BG/NR,
    adjacent-band pairing.
  - NDVI, FAI, NDSSI: standard, independently well-cited named indices.
  - NDWI, MNDWI: McFeeters (1996) and Xu (2006) respectively.

Reciprocal ratios, 2-band-sum ratios, and other ad-hoc combinations
without a specific citation are not included as candidates - they added
volume to correlation pruning without a corresponding justification for
why that particular combination should matter.
"""
import numpy as np
import pandas as pd

BASE_BANDS = ["red_corr7", "green_corr7", "blue_corr7", "nir_corr7",
              "swir1_corr7", "swir2_corr7", "temp_corr7"]

# not candidate features: row identifiers, join keys, match-quality
# metadata (time_diff doesn't exist at deployment time for an arbitrary
# satellite pass with no paired field sample), and the CV/holdout scaffolding
# columns spatial_cv.build_splits() adds
NON_FEATURE_COLS = {
    "siteSR_id", "date", "HUC4", "HUC8", "harmonized_value", "mission",
    "misc_flag", "lat", "lon", "time_diff",
    "holdout_part", "is_holdout",
}


def add_spectral_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    r, g, b = df["red_corr7"], df["green_corr7"], df["blue_corr7"]
    n, s1, s2 = df["nir_corr7"], df["swir1_corr7"], df["swir2_corr7"]

    df["BR"] = b / r
    df["BG"] = b / g
    df["NR"] = n / r
    df["GR"] = g / r

    df["fai"] = n - (r + (s1 - r) * ((830 - 660) / (1650 -660)))
    df["NDVI"] = (n - r) / (n + r)
    df["NDSSI"] = (b - n) / (b + n)
    df["NDWI"] = (g - n) / (g + n)     # McFeeters 1996
    df["MNDWI"] = (g - s1) / (g + s1)  # Xu 2006

    index_cols = ["BR", "BG", "NR", "GR", "fai", "NDVI", "NDSSI", "NDWI", "MNDWI"]
    # xgboost/lightgbm handle NaN natively; a ratio landing on inf (band==0
    # denominator) is recoded to NaN so it's treated as missing rather than
    # as an extreme value
    df[index_cols] = df[index_cols].replace([np.inf, -np.inf], np.nan)

    return df


SITE_COARSEN_PCT_COLS = ["pct_impervious_2006", "pct_urban_2006", "pct_forest_2006",
                         "pct_cropland_2006", "pct_wetland_2006"]

# 99th percentile of catchment_area_sqkm across the ~564 distinct waterbodies
# in the training data (fixed constant, not recomputed per call - a
# winsorization threshold has to be fit once on the reference distribution
# and applied identically everywhere, training and any later out-of-sample
# application alike, the same way a fitted scaler would be). At this cutoff,
# Lake Powell (2,667 km2) collapses to the same capped value as 5 other real,
# well-known large lakes (Great Salt Lake, Lake Fort Peck, Flathead Lake,
# Utah Lake, and one unnamed waterbody at 20,188 km2) - only 6 of 564
# waterbodies are touched at all.
CATCHMENT_WINSORIZE_CAP_SQKM = 617.1


def coarsen_site_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce catchment/land-cover feature precision and extremity so a
    heavily-resampled or catchment-outlier waterbody's static LakeCat
    values - identical for every site and every date on that waterbody,
    since they're joined once per NHDPlusV2 comid (see
    02b_pull_site_characteristics.Rmd) - can't act as a de facto waterbody ID
    for the model to key a memorized SDD value off of ("this is Pathfinder,
    therefore SDD = X"; "this is Lake Powell, therefore SDD = Y").

    catchment_area_sqkm: winsorized at CATCHMENT_WINSORIZE_CAP_SQKM (the
    training-data 99th percentile), *then* log-scale binned (round log10 to
    1 decimal, ~26% multiplicative bands). These fix two different problems
    and one doesn't substitute for the other. Log-binning alone reduces
    precision but preserves relative distance - it does nothing for a
    waterbody with no real neighbors in the distribution, which is exactly
    Lake Powell's situation: at 2,667 km2 (99.5th percentile by waterbody)
    it remains cleanly separated from every other large reservoir even
    after log-binning, still usable as a near-unique fingerprint, and SHAP
    analysis on a 10-seed ensemble confirmed the model was leaning on it
    there specifically (r=0.49 between per-HUC8 catchment-SHAP and
    per-HUC8 RMSE across the holdout; Lake Powell's own HUC8 was the single
    largest outlier in both). Winsorizing first removes the isolation
    itself, forcing Powell to collide with its nearest real peers, before
    log-binning further reduces precision on top of that. Chosen over
    dropping the feature outright: catchment_area_sqkm is unanimously
    selected by every seed across every re-run of this pipeline, and has
    independent literature support (Martinsen & Sand-Jensen 2022; Al-Shaibah
    et al. 2026) for carrying real signal beyond what the spectral bands
    capture - the isolation, not the feature, was the problem.

    Multi-HUC8 waterbodies more broadly (Lake Powell included) still
    straddle CV/holdout boundaries the same as before - that's a
    partitioning question (project memory partition-sensitivity-findings),
    not something this function addresses or is meant to.

    pct_*_2006 land-cover fractions: rounded to the nearest percentage
    point.
    """
    df = df.copy()

    valid = df["catchment_area_sqkm"].notna() & (df["catchment_area_sqkm"] > 0)
    capped = df.loc[valid, "catchment_area_sqkm"].clip(upper=CATCHMENT_WINSORIZE_CAP_SQKM)
    df.loc[valid, "catchment_area_sqkm"] = 10 ** np.log10(capped).round(1)

    df[SITE_COARSEN_PCT_COLS] = df[SITE_COARSEN_PCT_COLS].round(0)

    return df


def candidate_feature_list(df: pd.DataFrame) -> list:
    exclude = set(NON_FEATURE_COLS)
    exclude |= {c for c in df.columns if c.startswith("cvfold_seed")}
    return [c for c in df.columns if c not in exclude]


def correlation_prune(df: pd.DataFrame, feats: list, target: str = "harmonized_value",
                       corr_threshold: float = 0.95) -> list:
    """Cluster mutually-redundant (|r| > threshold) features via connected
    components (union-find) and keep the one most correlated with the
    target in each cluster."""
    corr_mat = df[feats].corr(method="pearson").abs()
    n = len(feats)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    idx = {f: i for i, f in enumerate(feats)}
    for i in range(n):
        for j in range(i + 1, n):
            if corr_mat.iloc[i, j] > corr_threshold:
                union(i, j)

    target_corr = df[feats].apply(lambda col: abs(col.corr(df[target])))

    clusters = {}
    for f in feats:
        root = find(idx[f])
        clusters.setdefault(root, []).append(f)

    kept = []
    for members in clusters.values():
        best = max(members, key=lambda f: target_corr[f] if pd.notna(target_corr[f]) else -1)
        kept.append(best)
    return kept
