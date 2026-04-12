"""Cross-validation utilities for model evaluation."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np


class KFoldCV:
    """K-fold cross-validation splitter that yields index arrays.

    The splitter returns ``(train_indices, val_indices)`` pairs so callers
    can slice any aligned arrays they need. This keeps the splitter reusable
    for model training, preprocessing, and search utilities.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: int | None = None,
    ) -> None:
        if not isinstance(n_splits, int):
            raise TypeError("n_splits must be an integer.")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield train and validation index arrays for each fold."""
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if X.ndim == 0:
            raise ValueError("X must contain at least one sample.")

        n_samples = X.shape[0]
        if self.n_splits > n_samples:
            raise ValueError("n_splits cannot exceed the number of samples.")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError("y must be a numpy array.")
            if y.shape[0] != n_samples:
                raise ValueError("X and y must have the same number of samples.")

        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start = current
            stop = current + int(fold_size)
            val_indices = indices[start:stop]
            train_indices = np.concatenate((indices[:start], indices[stop:]))
            yield train_indices, val_indices
            current = stop


def _clone_estimator(model: Any) -> Any:
    """Return a fresh estimator instance for a single fold."""
    get_params = getattr(model, "get_params", None)
    if callable(get_params):
        params = get_params()
        try:
            return model.__class__(**params)
        except TypeError:
            pass

    return copy.deepcopy(model)


def cross_val_score(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFoldCV,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Run a full cross-validation loop and return one score per fold."""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array.")
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    scores: list[float] = []
    for train_indices, val_indices in cv.split(X, y):
        estimator = _clone_estimator(model)
        estimator.fit(X[train_indices], y[train_indices])
        predictions = estimator.predict(X[val_indices])
        scores.append(float(metric_fn(y[val_indices], predictions)))

    return np.asarray(scores, dtype=float)
