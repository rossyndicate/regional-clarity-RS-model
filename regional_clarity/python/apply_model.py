"""Apply an SDD ensemble and its applicability checks.

`model_dir` is a model folder: a download of one of the Hugging Face model
repositories, or `xg_models/v3_production/` (production) or
`xg_models/v3_band_only/` (band-only) in this repository. Predict and screen
with the same folder; each model's applicability files are its own.

`X` is one row per observation with every model input already engineered,
as returned by `prepare_model_inputs()` in `R/prepare_model_inputs.R` (with
`site_and_weather = FALSE` for the band-only model), or, for inputs built in
Python, after `add_spectral_indices()` and `coarsen_site_features()` from
`features.py`.

    from apply_model import load_ensemble, predict_sdd, check_applicability

    members = load_ensemble(model_dir)
    X["pred_sdd"] = predict_sdd(members, X)
    flags = check_applicability(X, model_dir, pred_sdd=X["pred_sdd"])
    in_domain = X[flags["pass_all"]]

The checks mirror steps 07 and 07b (07_regional_application.Rmd and
07b_regional_application_band_only.Rmd), which write the `applicability/`
files this module reads. Scene QA (check 1 in the model cards) has to be
applied upstream and is not repeated here.

A failed check flags an observation as outside the conditions the model was
trained on, where its estimate is an extrapolation and may be poor; it does
not by itself make the estimate unusable. `di` and `out_of_range` say how far
outside, so users can set their own cutoff.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
import xgboost as xgb
from scipy.spatial import cKDTree


def load_ensemble(model_dir):
    """Return the 40 (features, booster) members: 10 seeds x 4 CV folds."""
    members = []
    for seed_dir in sorted(Path(model_dir).glob("seed*")):
        feats = json.loads((seed_dir / "backward_elim_summary.json").read_text())["final_features"]
        for fold in range(1, 5):
            booster = xgb.Booster()
            booster.load_model(str(seed_dir / f"xgboost_fold{fold}.json"))
            members.append((feats, booster))
    return members


def predict_sdd(members, X):
    """Ensemble-mean SDD (m); each booster gets its own seed's features."""
    return np.mean([b.predict(xgb.DMatrix(X[feats])) for feats, b in members], axis=0)


def load_applicability(model_dir):
    """Read the AOA parameters, the training reference in AOA space, and the
    training region footprint."""
    app_dir = Path(model_dir) / "applicability"
    params = json.loads((app_dir / "aoa_parameters.json").read_text())
    reference = pd.read_csv(app_dir / "aoa_reference.csv")
    region = shapely.from_geojson((app_dir / "training_region.geojson").read_text())
    return params, reference, region


def in_training_region(X, region):
    """True where the site (WGS84_Longitude, WGS84_Latitude) falls inside the
    training region: the HUC4 basins scoped in 01_filter_AquaMatch_Data.Rmd,
    simplified to 30 m. Points on the boundary count as inside, as in 01's
    spatial join."""
    region = shapely.union_all(shapely.get_parts(region))
    return shapely.intersects_xy(region, X["WGS84_Longitude"].to_numpy(float),
                                 X["WGS84_Latitude"].to_numpy(float))


def dissimilarity_index(X, params, reference):
    """Meyer & Pebesma (2021) DI: weighted, standardized distance to the
    nearest training observation over the mean training distance. NaN where
    any AOA feature is missing."""
    aoa = pd.DataFrame(params["aoa_features"])
    Z = ((X[aoa["feature"]].to_numpy(float) - aoa["center"].to_numpy())
         / aoa["scale"].to_numpy() * aoa["weight"].to_numpy())
    ref = reference[aoa["feature"]].to_numpy(float)

    di = np.full(len(Z), np.nan)
    ok = ~np.isnan(Z).any(axis=1)
    di[ok] = cKDTree(ref).query(Z[ok], k=1)[0] / params["d_bar"]
    return di


def check_applicability(X, model_dir, pred_sdd=None, aoa_threshold=None):
    """Per-row pass/fail for each step 07 check, applied cumulatively, plus
    how far each row sits from the training data.

    `X` needs `WGS84_Latitude`, `WGS84_Longitude`, `date`, and `mission`
    (LT04, LT05, LE07, LC08, LC09) columns plus every model input. Pass
    `pred_sdd` to include the plausibility check; otherwise `pass_all` stops
    at the AOA check. `aoa_threshold` overrides the DI threshold in
    `aoa_parameters.json`.

    `di` is returned for every row with all AOA features present, whether or
    not it passes; `out_of_range` names the inputs outside their training
    range ("" when none).
    """
    params, reference, region = load_applicability(model_dir)
    if aoa_threshold is None:
        aoa_threshold = params["aoa_threshold"]
    ranges = pd.DataFrame(params["feature_ranges"])
    doy = pd.to_datetime(X["date"]).dt.dayofyear
    lo, hi = params["season_window_doy"]

    out = pd.DataFrame(index=X.index)
    out["pass_region"] = in_training_region(X, region)
    out["pass_mission"] = out["pass_region"] & X["mission"].isin(params["training_missions"])
    out["pass_season"] = out["pass_mission"] & doy.between(lo, hi)
    out["pass_complete"] = out["pass_season"] & X[ranges["feature"]].notna().all(axis=1)
    outside = pd.DataFrame({f: ~X[f].between(fmin, fmax) & X[f].notna()
                            for f, fmin, fmax in ranges[["feature", "min", "max"]].itertuples(index=False)},
                           index=X.index)
    names = np.full(len(X), "", dtype=object)
    for f in outside.columns:
        names = names + np.where(outside[f].to_numpy(), f + ",", "")
    out["out_of_range"] = pd.Series(names, index=X.index, dtype=object).str.rstrip(",")
    out["pass_range"] = out["pass_complete"] & ~outside.any(axis=1)
    out["di"] = dissimilarity_index(X, params, reference)
    out["pass_aoa"] = out["pass_range"] & (out["di"] <= aoa_threshold)
    if pred_sdd is not None:
        out["pass_plausible"] = out["pass_aoa"] & (np.asarray(pred_sdd) >= params["min_training_sdd_m"])
        out["pass_all"] = out["pass_plausible"]
    else:
        out["pass_all"] = out["pass_aoa"]
    return out
