"""Gap-aware hyperparameter selection: random search that picks, among
near-optimal trials, the one with the smallest train/validation RMSE gap
rather than always the single lowest validation RMSE.

Applied at every tuning stage in the production pipeline (04_make_models),
closing a long-deferred gap where earlier passes of this project used
gap-aware selection only for final tuning. Testing it earlier too (project
memory "workflow-v3-findings") cost nothing in aggregate accuracy and
bought real cross-fold feature-selection consistency (all three of
shore_flag/swir1_corr7/tmax_degC_prev30 went from 9/10 to 10/10 unanimous
across seeds x models).
"""
import random

import numpy as np

import metrics as M


def gap_aware_tune(mod, folds, feats, target, n_trials, seed, val_tolerance=0.02):
    rng = random.Random(seed)
    trials = []
    for _ in range(n_trials):
        params = mod._sample_params(rng)
        fold_train_rmse, fold_val_rmse = [], []
        for fold in folds:
            one_fold_model = mod.train_fold_models([fold], feats, target, params)
            train_pred = mod.predict_ensemble(one_fold_model, fold.train[feats])
            val_pred = mod.predict_ensemble(one_fold_model, fold.val[feats])
            fold_train_rmse.append(M.rmse(fold.train[target].values, train_pred))
            fold_val_rmse.append(M.rmse(fold.val[target].values, val_pred))
        mean_val = float(np.mean(fold_val_rmse))
        mean_train = float(np.mean(fold_train_rmse))
        trials.append(dict(params=params, mean_val_rmse=mean_val, mean_train_rmse=mean_train,
                            mean_gap=mean_val - mean_train))
    best_val = min(t["mean_val_rmse"] for t in trials)
    band = [t for t in trials if t["mean_val_rmse"] <= best_val * (1 + val_tolerance)]
    chosen = min(band, key=lambda t: t["mean_gap"])
    return dict(best_params=chosen["params"], best_score=chosen["mean_val_rmse"],
                best_gap=chosen["mean_gap"], best_train_rmse=chosen["mean_train_rmse"],
                pure_best_val_rmse=best_val, band_size=len(band), n_trials=n_trials)
