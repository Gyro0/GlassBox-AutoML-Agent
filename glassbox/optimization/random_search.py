"""Random search utilities for hyperparameter tuning."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from glassbox.optimization.cross_validation import KFoldCV, cross_val_score


class RandomSearchCV:
    """Random hyperparameter search over discrete parameter distributions."""

    def __init__(
        self,
        model_class: type,
        param_distributions: dict[str, list[Any]],
        n_iter: int,
        time_budget_seconds: int,
        cv: KFoldCV,
        scoring_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        if not isinstance(n_iter, int):
            raise TypeError("n_iter must be an integer.")
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
        if time_budget_seconds < 0:
            raise ValueError("time_budget_seconds cannot be negative.")

        self.model_class = model_class
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.time_budget_seconds = time_budget_seconds
        self.cv = cv
        self.scoring_fn = scoring_fn
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.best_model_: object | None = None
        self.results_: list[tuple[float, dict[str, Any]]] = []

    @staticmethod
    def _prepare_param_distributions(
        param_distributions: dict[str, list[Any]],
    ) -> dict[str, list[Any]]:
        """Normalize and validate parameter distributions."""
        prepared: dict[str, list[Any]] = {}
        for key, values in param_distributions.items():
            if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
                raise TypeError(
                    f"Parameter distributions for '{key}' must be iterable."
                )

            options = list(values)
            if len(options) == 0:
                raise ValueError(
                    f"Parameter distribution for '{key}' must contain at least one value."
                )
            prepared[key] = options

        return prepared

    @staticmethod
    def _sample_params(
        rng: np.random.Generator,
        param_distributions: dict[str, list[Any]],
    ) -> dict[str, Any]:
        """Draw one random parameter set from the given distributions."""
        sampled: dict[str, Any] = {}
        for key, options in param_distributions.items():
            sampled[key] = options[int(rng.integers(len(options)))]
        return sampled

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomSearchCV":
        """Evaluate random parameter samples and keep the best one."""
        prepared = self._prepare_param_distributions(self.param_distributions)
        rng = np.random.default_rng()
        max_iterations = 1 if not prepared else self.n_iter

        results: list[tuple[float, dict[str, Any]]] = []
        best_score = -np.inf
        best_params: dict[str, Any] | None = None

        start_time = time.perf_counter()
        for _ in range(max_iterations):
            params = self._sample_params(rng, prepared) if prepared else {}
            model = self.model_class(**params)
            fold_scores = cross_val_score(model, X, y, self.cv, self.scoring_fn)
            mean_score = float(np.mean(fold_scores))
            params_copy = dict(params)

            results.append((mean_score, params_copy))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params_copy

            if time.perf_counter() - start_time >= self.time_budget_seconds:
                break

        results.sort(key=lambda item: item[0], reverse=True)
        self.results_ = results

        if best_params is None:
            raise RuntimeError("Random search could not evaluate any parameter sets.")

        self.best_params_ = best_params
        self.best_score_ = float(best_score)
        self.best_model_ = self.model_class(**best_params)
        self.best_model_.fit(X, y)
        return self
