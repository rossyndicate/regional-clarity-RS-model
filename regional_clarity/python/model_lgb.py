"""LightGBM training, same spatial CV / random-search discipline as
model_xgb.py, so the two boosted-tree models get a comparably-sized
hyperparameter search."""
import random

import numpy as np
import lightgbm as lgb

PARAM_SPACE = dict(
    learning_rate=[0.01, 0.03, 0.05],
    num_leaves=[7, 15, 31],
    min_child_samples=[20, 30, 50],
    subsample=[0.6, 0.7, 0.8],
    colsample_bytree=[0.4, 0.5, 0.7],
    reg_alpha=[0, 0.1, 1, 5],
    reg_lambda=[1, 5, 10, 20],
)


def _sample_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in PARAM_SPACE.items()}


def _fit_fold(train_X, train_y, val_X, val_y, params, nrounds=3000, early_stop=100, train_weight=None):
    dtrain = lgb.Dataset(train_X, label=train_y, weight=train_weight)
    dval = lgb.Dataset(val_X, label=val_y, reference=dtrain)
    full_params = dict(objective="regression", metric="rmse", verbosity=-1, **params)
    booster = lgb.train(full_params, dtrain, num_boost_round=nrounds,
                         valid_sets=[dval],
                         callbacks=[lgb.early_stopping(early_stop, verbose=False)])
    return booster, booster.best_score["valid_0"]["rmse"]


def tune(folds: list, feats: list, target: str, n_trials: int = 25, seed: int = 47) -> dict:
    rng = random.Random(seed)
    best_params, best_score = None, np.inf
    trials = []
    for _ in range(n_trials):
        params = _sample_params(rng)
        fold_scores = []
        for fold in folds:
            _, val_rmse = _fit_fold(fold.train[feats], fold.train[target],
                                     fold.val[feats], fold.val[target], params)
            fold_scores.append(val_rmse)
        mean_score = float(np.mean(fold_scores))
        trials.append({**params, "mean_val_rmse": mean_score})
        if mean_score < best_score:
            best_score, best_params = mean_score, params
    return {"best_params": best_params, "best_score": best_score, "trials": trials}


def train_fold_models(folds: list, feats: list, target: str, params: dict, weight_fn=None) -> list:
    models = []
    for fold in folds:
        w = weight_fn(fold.train[target].values) if weight_fn is not None else None
        booster, _ = _fit_fold(fold.train[feats], fold.train[target],
                                fold.val[feats], fold.val[target], params,
                                nrounds=5000, early_stop=250, train_weight=w)
        models.append(booster)
    return models


def predict_ensemble(models: list, X) -> np.ndarray:
    preds = np.column_stack([m.predict(X, num_iteration=m.best_iteration) for m in models])
    return preds.mean(axis=1)
