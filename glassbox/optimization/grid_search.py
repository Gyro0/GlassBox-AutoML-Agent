"""Grid search utilities for hyperparameter tuning."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

import numpy as np

from glassbox.optimization.cross_validation import KFoldCV, cross_val_score


class GridSearchCV:
    """Exhaustive grid search over a discrete parameter space."""

    def __init__(
        self,
        model_class: type,
        param_grid: dict[str, list[Any]],
        cv: KFoldCV,
        scoring_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring_fn = scoring_fn
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.best_model_: object | None = None
        self.results_: list[tuple[float, dict[str, Any]]] = []

    @staticmethod
    def _expand_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
        """Expand a parameter grid into concrete constructor dictionaries."""
        if not param_grid:
            return [{}]

        keys = list(param_grid.keys())
        value_lists: list[list[Any]] = []

        for key in keys:
            values = param_grid[key]
            if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
                raise TypeError(
                    f"Parameter grid values for '{key}' must be iterable."
                )

            options = list(values)
            if len(options) == 0:
                raise ValueError(
                    f"Parameter grid for '{key}' must contain at least one value."
                )
            value_lists.append(options)

        combinations: list[dict[str, Any]] = []
        for combination in product(*value_lists):
            combinations.append(dict(zip(keys, combination, strict=True)))
        return combinations

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridSearchCV":
        """Evaluate every parameter combination and keep the best one."""
        candidates = self._expand_param_grid(self.param_grid)
        results: list[tuple[float, dict[str, Any]]] = []

        best_score = -np.inf
        best_params: dict[str, Any] | None = None

        for params in candidates:
            model = self.model_class(**params)
            fold_scores = cross_val_score(model, X, y, self.cv, self.scoring_fn)
            mean_score = float(np.mean(fold_scores))
            params_copy = dict(params)

            results.append((mean_score, params_copy))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params_copy

        results.sort(key=lambda item: item[0], reverse=True)
        self.results_ = results

        if best_params is None:
            raise RuntimeError("Grid search could not evaluate any parameter sets.")

        self.best_params_ = best_params
        self.best_score_ = float(best_score)
        self.best_model_ = self.model_class(**best_params)
        self.best_model_.fit(X, y)
        return self
