"""Numerical column profiling utilities."""

from __future__ import annotations

from collections import Counter

import numpy as np


def _mode(values: np.ndarray) -> float:
    counts = Counter(values.tolist())
    max_count = max(counts.values())
    # Deterministic tie-breaker: smallest value among equally frequent candidates.
    return float(min(value for value, count in counts.items() if count == max_count))


def _skewness(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if np.isclose(std, 0.0):
        return 0.0
    z_scores = (values - mean) / std
    return float(np.mean(z_scores**3))


def _kurtosis(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if np.isclose(std, 0.0):
        return -3.0
    z_scores = (values - mean) / std
    return float(np.mean(z_scores**4) - 3.0)


def profile_numeric_columns(
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> dict[int | str, dict[str, float]]:
    """Return descriptive statistics for each numerical column in X."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D NumPy array.")

    if feature_names is not None and len(feature_names) != X.shape[1]:
        raise ValueError("feature_names length must match X.shape[1].")

    profile: dict[int | str, dict[str, float]] = {}
    for col_idx in range(X.shape[1]):
        column = X[:, col_idx].astype(float)
        key: int | str = feature_names[col_idx] if feature_names is not None else col_idx
        profile[key] = {
            "mean": float(np.mean(column)),
            "median": float(np.median(column)),
            "mode": _mode(column),
            "std": float(np.std(column)),
            "variance": float(np.var(column)),
            "skewness": _skewness(column),
            "kurtosis": _kurtosis(column),
        }

    return profile