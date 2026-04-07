"""Regression metric functions implemented from core formulas."""

from __future__ import annotations

import numpy as np

_SHAPE_ERROR_MESSAGE = "y_true and y_pred must have the same number of samples."


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MAE = (1/n) * sum(|y_i - y_hat_i|)."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(_SHAPE_ERROR_MESSAGE)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MSE = (1/n) * sum((y_i - y_hat_i)^2)."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(_SHAPE_ERROR_MESSAGE)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R2 = 1 - sum((y_i - y_hat_i)^2) / sum((y_i - y_bar)^2)."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(_SHAPE_ERROR_MESSAGE)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))

    if np.isclose(ss_tot, 0.0):
        return 0.0
    return 1.0 - (ss_res / ss_tot)