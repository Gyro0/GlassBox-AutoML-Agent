"""Small NumPy matrix helpers shared across modules."""

from __future__ import annotations

import numpy as np


def dot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return matrix multiplication of A and B."""
    return np.matmul(A, B)


def transpose(A: np.ndarray) -> np.ndarray:
    """Return the transpose of A."""
    return np.transpose(A)


def column_mean(X: np.ndarray) -> np.ndarray:
    """Return column-wise mean values for a 2D array."""
    return np.mean(X, axis=0)


def column_variance(X: np.ndarray) -> np.ndarray:
    """Return column-wise population variance values for a 2D array."""
    return np.var(X, axis=0)


def column_std(X: np.ndarray) -> np.ndarray:
    """Return column-wise population standard deviation values for a 2D array."""
    return np.std(X, axis=0)