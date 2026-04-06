"""Random search utilities for hyperparameter tuning."""

from typing import Any, Callable

import numpy as np

from glassbox.optimization.cross_validation import KFoldCV


class RandomSearchCV:
    """Placeholder random search interface."""

    def __init__(
        self,
        model_class: type,
        param_distributions: dict[str, list[Any]],
        n_iter: int,
        time_budget_seconds: int,
        cv: KFoldCV,
        scoring_fn: Callable[[np.ndarray, np.ndarray], float],
    ):
        self.model_class = model_class
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.time_budget_seconds = time_budget_seconds
        self.cv = cv
        self.scoring_fn = scoring_fn
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.best_model_: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomSearchCV":
        """Run placeholder random search over sampled parameters."""
        raise NotImplementedError("RandomSearchCV.fit is not implemented yet.")
