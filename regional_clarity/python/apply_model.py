"""Apply the production SDD ensemble and its applicability checks.

`model_dir` is the production model folder: a download of the Hugging Face
model repository, or `xg_models/v3_production/` in this repository. `X` is
one row per observation with every model input already engineered, as
returned by `prepare_model_inputs()` in `R/prepare_model_inputs.R` (or, for
inputs built in Python, after `add_spectral_indices()` and
`coarsen_site_features()` from `features.py`).

    from apply_model import load_ensemble, predict_sdd, check_applicability

    members = load_ensemble(model_dir)
    X["pred_sdd"] = predict_sdd(members, X)
    flags = check_applicability(X, model_dir, pred_sdd=X["pred_sdd"])
    usable = X[flags["pass_all"]]

The checks mirror step 07 (07_regional_application.Rmd), which writes the
`applicability/` files this module reads. Scene QA (check 1 in the model
card) has to be applied upstream and is not repeated here.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
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
    """Read the AOA parameters and the training reference in AOA space."""
    app_dir = Path(model_dir) / "applicability"
    params = json.loads((app_dir / "aoa_parameters.json").read_text())
    reference = pd.read_csv(app_dir / "aoa_reference.csv")
    return params, reference


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


def check_applicability(X, model_dir, pred_sdd=None):
    """Per-row pass/fail for each step 07 check, applied cumulatively.

    `X` needs `date` and `mission` (LT04, LT05, LE07, LC08, LC09) columns
    plus every model input. Pass `pred_sdd` to include the plausibility
    check; otherwise `pass_all` stops at the AOA check.
    """
    params, reference = load_applicability(model_dir)
    ranges = pd.DataFrame(params["feature_ranges"])
    doy = pd.to_datetime(X["date"]).dt.dayofyear
    lo, hi = params["season_window_doy"]

    out = pd.DataFrame(index=X.index)
    out["pass_mission"] = X["mission"].isin(params["training_missions"])
    out["pass_season"] = out["pass_mission"] & doy.between(lo, hi)
    out["pass_complete"] = out["pass_season"] & X[ranges["feature"]].notna().all(axis=1)
    in_range = np.ones(len(X), dtype=bool)
    for f, fmin, fmax in ranges[["feature", "min", "max"]].itertuples(index=False):
        in_range &= X[f].between(fmin, fmax).to_numpy()
    out["pass_range"] = out["pass_complete"] & in_range
    out["di"] = dissimilarity_index(X, params, reference)
    out["pass_aoa"] = out["pass_range"] & (out["di"] <= params["aoa_threshold"])
    if pred_sdd is not None:
        out["pass_plausible"] = out["pass_aoa"] & (np.asarray(pred_sdd) >= params["min_training_sdd_m"])
        out["pass_all"] = out["pass_plausible"]
    else:
        out["pass_all"] = out["pass_aoa"]
    return out
