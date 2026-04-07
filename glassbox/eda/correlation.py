"""Correlation analysis helpers."""

from __future__ import annotations

import numpy as np


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    numerator = float(np.sum(x_centered * y_centered))
    denominator = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))

    if np.isclose(denominator, 0.0):
        return 0.0
    return numerator / denominator


def build_pearson_correlation_matrix(
    X: np.ndarray,
    feature_names: list[str] | None = None,
    collinearity_threshold: float = 0.85,
) -> tuple[np.ndarray, list[str], list[tuple[str, str, float]]]:
    """Build correlation matrix and return highly collinear feature pairs."""
    if X.ndim != 2:
        raise ValueError("X must be a 2D NumPy array.")

    n_features = X.shape[1]
    names = feature_names if feature_names is not None else [f"feature_{i}" for i in range(n_features)]
    if len(names) != n_features:
        raise ValueError("feature_names length must match X.shape[1].")

    corr_matrix = np.zeros((n_features, n_features), dtype=float)
    for i in range(n_features):
        for j in range(n_features):
            corr_matrix[i, j] = _pearson_correlation(X[:, i].astype(float), X[:, j].astype(float))

    high_collinearity: list[tuple[str, str, float]] = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            r_value = float(corr_matrix[i, j])
            if abs(r_value) > collinearity_threshold:
                high_collinearity.append((names[i], names[j], r_value))

    return corr_matrix, names, high_collinearity