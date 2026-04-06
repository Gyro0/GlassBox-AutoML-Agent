"""Grid search utilities for hyperparameter tuning."""

from typing import Any, Callable

import numpy as np

from glassbox.optimization.cross_validation import KFoldCV


class GridSearchCV:
    """Placeholder grid search interface."""

    def __init__(
        self,
        model_class: type,
        param_grid: dict[str, list[Any]],
        cv: KFoldCV,
        scoring_fn: Callable[[np.ndarray, np.ndarray], float],
    ):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring_fn = scoring_fn
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.best_model_: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GridSearchCV":
        """Run placeholder grid search over parameter combinations."""
        raise NotImplementedError("GridSearchCV.fit is not implemented yet.")
