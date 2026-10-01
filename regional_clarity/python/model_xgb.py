"""XGBoost training over the HUC8-random spatial CV (regional_clarity.python.spatial_cv).
Hyperparameter search is a random search over a reasonable grid, scored by
mean CV RMSE - an automated stand-in for the original 04_make_models.Rmd's
manual grid search, gap-aware-selected by regional_clarity.python.tuning
rather than taken on raw CV RMSE alone."""
import random

import numpy as np
import xgboost as xgb

PARAM_SPACE = dict(
    # shallow, regularized trees - appropriate once the feature set is down
    # to a small, correlation-pruned/backward-eliminated core
    eta=[0.01, 0.03, 0.05],
    max_depth=[2, 3, 4],
    min_child_weight=[3, 5, 7, 10],
    subsample=[0.6, 0.7, 0.8],
    colsample_bytree=[0.4, 0.5, 0.7],
    reg_alpha=[0, 0.1, 1, 5],
    reg_lambda=[1, 5, 10, 20],
)


def _sample_params(rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in PARAM_SPACE.items()}


def _fit_fold(train_X, train_y, val_X, val_y, params, nrounds=3000, early_stop=100, train_weight=None):
    dtrain = xgb.DMatrix(train_X, label=train_y, weight=train_weight)
    dval = xgb.DMatrix(val_X, label=val_y)
    full_params = dict(booster="gbtree", objective="reg:squarederror", **params)
    booster = xgb.train(full_params, dtrain, num_boost_round=nrounds,
                         evals=[(dtrain, "train"), (dval, "val")],
                         early_stopping_rounds=early_stop, verbose_eval=False)
    return booster, booster.best_score


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
    d = xgb.DMatrix(X)
    preds = np.column_stack([m.predict(d) for m in models])
    return preds.mean(axis=1)
