"""Regression metrics matching the R {Metrics} package definitions used in
the original 04_make_models.Rmd, so results stay comparable to that
baseline."""
import numpy as np


def _clean(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[ok], y_pred[ok]


def rmse(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


def bias(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(y_pred - y_true))


def percent_bias(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(100 * np.sum(y_pred - y_true) / np.sum(y_true))


def smape(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))))


def r2(y_true, y_pred):
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)


def all_metrics(y_true, y_pred) -> dict:
    return dict(rmse=rmse(y_true, y_pred), mae=mae(y_true, y_pred),
                mape=mape(y_true, y_pred), bias=bias(y_true, y_pred),
                p_bias=percent_bias(y_true, y_pred), smape=smape(y_true, y_pred),
                r2=r2(y_true, y_pred), n=int(len(np.asarray(y_true))))
