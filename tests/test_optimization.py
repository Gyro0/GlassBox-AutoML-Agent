"""Unit tests for glassbox.optimization."""

from __future__ import annotations

import numpy as np
import pytest
import glassbox.optimization.random_search as random_search_module

from glassbox.evaluation.classification import classification_report
from glassbox.evaluation.regression import r2_score
from glassbox.models.linear import LinearRegression, LogisticRegression
from glassbox.optimization.cross_validation import KFoldCV, cross_val_score
from glassbox.optimization.grid_search import GridSearchCV
from glassbox.optimization.random_search import RandomSearchCV


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Small helper to extract accuracy for generic CV scoring."""
    return float(classification_report(y_true, y_pred)["accuracy"])


def test_kfold_split_covers_all_samples_without_leakage() -> None:
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    splits = list(KFoldCV(n_splits=3).split(X, y))

    assert len(splits) == 3

    validation_sizes: list[int] = []
    all_validation_indices: list[int] = []
    for train_indices, val_indices in splits:
        assert len(np.intersect1d(train_indices, val_indices)) == 0
        assert len(train_indices) + len(val_indices) == X.shape[0]
        validation_sizes.append(len(val_indices))
        all_validation_indices.extend(val_indices.tolist())

    assert validation_sizes == [4, 3, 3]
    assert sorted(all_validation_indices) == list(range(X.shape[0]))


def test_kfold_split_shuffle_is_reproducible() -> None:
    X = np.arange(24).reshape(12, 2)

    shuffled_a = list(KFoldCV(n_splits=4, shuffle=True, random_state=7).split(X))
    shuffled_b = list(KFoldCV(n_splits=4, shuffle=True, random_state=7).split(X))
    ordered = list(KFoldCV(n_splits=4, shuffle=False).split(X))

    for (train_a, val_a), (train_b, val_b) in zip(shuffled_a, shuffled_b):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)

    assert not np.array_equal(shuffled_a[0][1], ordered[0][1])


def test_kfold_rejects_invalid_split_configuration() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        KFoldCV(n_splits=1)

    cv = KFoldCV(n_splits=6)
    X = np.arange(10).reshape(5, 2)
    with pytest.raises(ValueError, match="cannot exceed"):
        list(cv.split(X))


def test_cross_val_score_regression_returns_high_scores() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = 4.0 * X[:, 0] - 1.5 * X[:, 1] + 2.0 + rng.normal(scale=0.05, size=120)

    model = LinearRegression(learning_rate=0.05, n_iterations=1500)
    scores = cross_val_score(model, X, y, KFoldCV(n_splits=5), r2_score)

    assert scores.shape == (5,)
    assert np.all(scores > 0.98)
    assert model._fitted is False


def test_cross_val_score_classification_works_with_custom_metric() -> None:
    rng = np.random.default_rng(0)
    X_pos = rng.normal(loc=2.0, scale=0.7, size=(50, 2))
    X_neg = rng.normal(loc=-2.0, scale=0.7, size=(50, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [0] * 50)

    permutation = rng.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    scores = cross_val_score(
        model,
        X,
        y,
        KFoldCV(n_splits=5, shuffle=True, random_state=42),
        _accuracy_score,
    )

    assert scores.shape == (5,)
    assert np.all(scores >= 0.90)


def test_cross_val_score_rejects_inconsistent_lengths() -> None:
    X = np.arange(12).reshape(6, 2)
    y = np.arange(5)
    model = LinearRegression()

    with pytest.raises(ValueError, match="same number of samples"):
        cross_val_score(model, X, y, KFoldCV(n_splits=3), r2_score)


def test_grid_search_finds_best_params_and_sorts_results() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = 4.0 * X[:, 0] - 1.5 * X[:, 1] + 2.0 + rng.normal(scale=0.05, size=120)

    search = GridSearchCV(
        model_class=LinearRegression,
        param_grid={
            "learning_rate": [0.001, 0.05],
            "n_iterations": [200, 1500],
            "tol": [1e-7],
        },
        cv=KFoldCV(n_splits=5),
        scoring_fn=r2_score,
    )

    result = search.fit(X, y)

    assert result is search
    assert len(search.results_) == 4
    assert search.best_params_ == search.results_[0][1]
    assert search.best_score_ == pytest.approx(search.results_[0][0])
    assert all(
        search.results_[idx][0] >= search.results_[idx + 1][0]
        for idx in range(len(search.results_) - 1)
    )
    assert isinstance(search.best_model_, LinearRegression)
    assert search.best_model_._fitted is True
    assert search.best_model_.predict(X[:5]).shape == (5,)


def test_grid_search_supports_classifier_scoring() -> None:
    rng = np.random.default_rng(0)
    X_pos = rng.normal(loc=2.0, scale=0.7, size=(50, 2))
    X_neg = rng.normal(loc=-2.0, scale=0.7, size=(50, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [0] * 50)

    permutation = rng.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]

    search = GridSearchCV(
        model_class=LogisticRegression,
        param_grid={
            "learning_rate": [0.01, 0.1],
            "n_iterations": [200, 800],
            "threshold": [0.5],
        },
        cv=KFoldCV(n_splits=5, shuffle=True, random_state=42),
        scoring_fn=_accuracy_score,
    )

    search.fit(X, y)

    assert len(search.results_) == 4
    assert search.best_score_ is not None
    assert search.best_score_ >= 0.90
    assert isinstance(search.best_model_, LogisticRegression)


def test_grid_search_supports_empty_param_grid() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = 3.0 * X[:, 0] + 0.5 * X[:, 1]

    search = GridSearchCV(
        model_class=LinearRegression,
        param_grid={},
        cv=KFoldCV(n_splits=3),
        scoring_fn=r2_score,
    )

    search.fit(X, y)

    assert search.results_ and len(search.results_) == 1
    assert search.best_params_ == {}
    assert search.best_model_ is not None


def test_grid_search_rejects_empty_parameter_options() -> None:
    search = GridSearchCV(
        model_class=LinearRegression,
        param_grid={"learning_rate": []},
        cv=KFoldCV(n_splits=3),
        scoring_fn=r2_score,
    )
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6, dtype=float)

    with pytest.raises(ValueError, match="at least one value"):
        search.fit(X, y)


def test_random_search_runs_requested_iterations_and_tracks_best_model() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = 4.0 * X[:, 0] - 1.5 * X[:, 1] + 2.0 + rng.normal(scale=0.05, size=120)

    search = RandomSearchCV(
        model_class=LinearRegression,
        param_distributions={
            "learning_rate": [0.05],
            "n_iterations": [1500],
            "tol": [1e-7],
        },
        n_iter=3,
        time_budget_seconds=30,
        cv=KFoldCV(n_splits=5),
        scoring_fn=r2_score,
    )

    result = search.fit(X, y)

    assert result is search
    assert len(search.results_) == 3
    assert search.best_params_ == search.results_[0][1]
    assert search.best_score_ == pytest.approx(search.results_[0][0])
    assert isinstance(search.best_model_, LinearRegression)
    assert search.best_model_._fitted is True
    assert search.best_model_.predict(X[:5]).shape == (5,)


def test_random_search_supports_classifier_scoring() -> None:
    rng = np.random.default_rng(0)
    X_pos = rng.normal(loc=2.0, scale=0.7, size=(50, 2))
    X_neg = rng.normal(loc=-2.0, scale=0.7, size=(50, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [0] * 50)

    permutation = rng.permutation(X.shape[0])
    X = X[permutation]
    y = y[permutation]

    search = RandomSearchCV(
        model_class=LogisticRegression,
        param_distributions={
            "learning_rate": [0.1],
            "n_iterations": [800],
            "threshold": [0.5],
        },
        n_iter=2,
        time_budget_seconds=30,
        cv=KFoldCV(n_splits=5, shuffle=True, random_state=42),
        scoring_fn=_accuracy_score,
    )

    search.fit(X, y)

    assert len(search.results_) == 2
    assert search.best_score_ is not None
    assert search.best_score_ >= 0.90
    assert isinstance(search.best_model_, LogisticRegression)


def test_random_search_supports_empty_param_distributions() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = 3.0 * X[:, 0] + 0.5 * X[:, 1]

    search = RandomSearchCV(
        model_class=LinearRegression,
        param_distributions={},
        n_iter=5,
        time_budget_seconds=30,
        cv=KFoldCV(n_splits=3),
        scoring_fn=r2_score,
    )

    search.fit(X, y)

    assert len(search.results_) == 1
    assert search.best_params_ == {}
    assert search.best_model_ is not None


def test_random_search_respects_time_budget(monkeypatch) -> None:
    timestamps = iter([0.0, 2.0])

    def fake_perf_counter() -> float:
        return next(timestamps)

    monkeypatch.setattr(random_search_module.time, "perf_counter", fake_perf_counter)

    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10, dtype=float)
    search = RandomSearchCV(
        model_class=LinearRegression,
        param_distributions={
            "learning_rate": [0.01],
            "n_iterations": [10],
            "tol": [1e-7],
        },
        n_iter=5,
        time_budget_seconds=1,
        cv=KFoldCV(n_splits=2),
        scoring_fn=r2_score,
    )

    search.fit(X, y)

    assert len(search.results_) == 1


def test_random_search_rejects_empty_parameter_options() -> None:
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6, dtype=float)
    search = RandomSearchCV(
        model_class=LinearRegression,
        param_distributions={"learning_rate": []},
        n_iter=3,
        time_budget_seconds=30,
        cv=KFoldCV(n_splits=3),
        scoring_fn=r2_score,
    )

    with pytest.raises(ValueError, match="at least one value"):
        search.fit(X, y)


def test_random_search_rejects_invalid_constructor_arguments() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RandomSearchCV(
            model_class=LinearRegression,
            param_distributions={"learning_rate": [0.01]},
            n_iter=0,
            time_budget_seconds=30,
            cv=KFoldCV(n_splits=3),
            scoring_fn=r2_score,
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        RandomSearchCV(
            model_class=LinearRegression,
            param_distributions={"learning_rate": [0.01]},
            n_iter=3,
            time_budget_seconds=-1,
            cv=KFoldCV(n_splits=3),
            scoring_fn=r2_score,
        )
