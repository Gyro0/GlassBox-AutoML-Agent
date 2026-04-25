"""Abstract base class for all GlassBox models.

All estimators inherit from BaseModel and must implement fit() and predict().
score() is provided with sensible defaults: accuracy for classifiers, R² for regressors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """Abstract base for all GlassBox estimators.

    Attributes
    ----------
    _fitted : bool
        Set to True after a successful fit() call.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Learn model parameters from training data.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y : np.ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        self
            The fitted model instance (allows method chaining).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for new data.
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        Returns
        -------
        np.ndarray, shape (n_samples,)
            Predicted values (class labels for classifiers, continuous
            values for regressors).
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy for classifiers, R² for regressors.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
            True target values.

        Returns
        -------
        float
        """
        self._check_is_fitted()
        y_pred = self.predict(X)

        if self._is_classifier():
            # Accuracy
            return float(np.mean(y_pred == y))

        # R² coefficient of determination
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return float(1.0 - ss_res / ss_tot)

    def _is_classifier(self) -> bool:
        """True if _task == 'classification'."""
        return getattr(self, "_task", "regression") == "classification"

    def _check_is_fitted(self) -> None:
        """Raise ``RuntimeError`` if the model has not been fitted yet."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit(X, y) before using this model."
            )

    @staticmethod
    def _validate_inputs(X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Run basic sanity checks on input arrays.
        Parameters
        ----------
        X : np.ndarray
            Must be a 2-D numeric array.
        y : np.ndarray or None
            If provided, must be 1-D with length equal to ``X.shape[0]``.
        Raises
        ------
        TypeError
            If inputs are not NumPy arrays.
        ValueError
            If shapes are inconsistent.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X).__name__}.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got {X.ndim}-D array.")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError(
                    f"y must be a numpy array, got {type(y).__name__}."
                )
            if y.ndim != 1:
                raise ValueError(f"y must be 1-D, got {y.ndim}-D array.")
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"X and y have inconsistent lengths: "
                    f"X has {X.shape[0]} samples, y has {y.shape[0]}."
                )

    def __repr__(self) -> str:
        """Readable string representation showing constructor parameters."""
        params = ", ".join(
            f"{k}={v!r}"
            for k, v in self.get_params().items()
        )
        return f"{self.__class__.__name__}({params})"

    def get_params(self) -> dict:
        """Return constructor parameters as a dict."""
        import inspect

        init_sig = inspect.signature(self.__init__)  # type: ignore[misc]
        return {
            name: getattr(self, name, None)
            for name in init_sig.parameters
            if name != "self"
        }
