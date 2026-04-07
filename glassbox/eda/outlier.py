"""Outlier detection and capping using IQR bounds."""

from __future__ import annotations

from typing import Literal

import numpy as np


def _percentile(sorted_values: np.ndarray, q: float) -> float:
    if sorted_values.size == 0:
        raise ValueError("Cannot compute percentile of an empty array.")

    position = (sorted_values.size - 1) * q
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    weight = position - lower_index
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    return lower_value + (upper_value - lower_value) * weight


def iqr_outlier_handler(
    X: np.ndarray,
    mode: Literal["flag", "cap"] = "flag",
    return_row_indices: bool = False,
) -> np.ndarray | list[int]:
    """Flag outliers or cap values based on per-column IQR bounds."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D NumPy array.")
    if mode not in {"flag", "cap"}:
        raise ValueError("mode must be either 'flag' or 'cap'.")

    values = X.astype(float)
    n_rows, n_cols = values.shape
    lower_bounds = np.zeros(n_cols, dtype=float)
    upper_bounds = np.zeros(n_cols, dtype=float)

    for col_idx in range(n_cols):
        col_sorted = np.sort(values[:, col_idx])
        q1 = _percentile(col_sorted, 0.25)
        q3 = _percentile(col_sorted, 0.75)
        iqr = q3 - q1
        lower_bounds[col_idx] = q1 - 1.5 * iqr
        upper_bounds[col_idx] = q3 + 1.5 * iqr

    if mode == "flag":
        mask = np.zeros((n_rows, n_cols), dtype=bool)
        for col_idx in range(n_cols):
            mask[:, col_idx] = (values[:, col_idx] < lower_bounds[col_idx]) | (values[:, col_idx] > upper_bounds[col_idx])
        if return_row_indices:
            return np.nonzero(np.any(mask, axis=1))[0].tolist()
        return mask

    capped = values.copy()
    for col_idx in range(n_cols):
        capped[:, col_idx] = np.clip(capped[:, col_idx], lower_bounds[col_idx], upper_bounds[col_idx])
    return capped