"""Compare GlassBox models with equivalent Scikit-Learn baselines.

Run from the repository root:

    python3 benchmarks/sklearn_comparison.py

Scikit-Learn is intentionally used only in this benchmark script, never in
the ``glassbox`` package itself.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glassbox.evaluation.classification import classification_report
from glassbox.evaluation.regression import r2_score
from glassbox.models import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GaussianNaiveBayes,
    KNearestNeighbors,
    LinearRegression,
    LogisticRegression,
    RandomForestClassifier,
    RandomForestRegressor,
)
from glassbox.preprocessing import SimpleImputer, StandardScaler


DIABETES_CSV = ROOT / "data" / "_uploaded.csv"
DIABETES_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
DIABETES_ZERO_AS_MISSING = {
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
}
CALIFORNIA_HOUSING_SAMPLE_SIZE = 5_000


def _require_sklearn() -> tuple[object, ...]:
    """Import Scikit-Learn lazily with a helpful message if absent."""
    try:
        from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
        from sklearn.linear_model import LinearRegression as SkLinearRegression
        from sklearn.linear_model import LogisticRegression as SkLogisticRegression
        from sklearn.metrics import accuracy_score, r2_score as sk_r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import GaussianNB as SkGaussianNB
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.datasets import fetch_california_housing
        from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
        from sklearn.tree import DecisionTreeRegressor as SkDecisionTreeRegressor
    except ImportError as exc:
        raise SystemExit(
            "Scikit-Learn is required for this benchmark. Install it with "
            "`python3 -m pip install scikit-learn`."
        ) from exc

    return (
        SkRandomForestClassifier,
        SkRandomForestRegressor,
        SkLinearRegression,
        SkLogisticRegression,
        accuracy_score,
        sk_r2_score,
        train_test_split,
        SkGaussianNB,
        KNeighborsClassifier,
        KNeighborsRegressor,
        fetch_california_housing,
        SkDecisionTreeClassifier,
        SkDecisionTreeRegressor,
    )


def _classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic binary classification benchmark dataset."""
    rng = np.random.default_rng(7)
    n_samples = 260
    X = rng.normal(size=(n_samples, 5))
    score = (
        1.6 * X[:, 0]
        - 1.2 * X[:, 1]
        + 0.8 * X[:, 2]
        + 0.5 * (X[:, 3] > 0.0)
        + rng.normal(scale=0.25, size=n_samples)
    )
    y = (score > np.median(score)).astype(int)
    return X, y


def _regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic regression benchmark dataset."""
    rng = np.random.default_rng(11)
    n_samples = 260
    X = rng.normal(size=(n_samples, 5))
    y = (
        4.0
        + 2.8 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.9 * X[:, 2]
        + rng.normal(scale=0.2, size=n_samples)
    )
    return X, y


def _diabetes_data(csv_path: Path = DIABETES_CSV) -> tuple[np.ndarray, np.ndarray]:
    """Load the diabetes CSV as a real binary classification benchmark."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"Diabetes benchmark CSV is empty: {csv_path}")

    X = np.empty((len(rows), len(DIABETES_FEATURES)), dtype=float)
    y = np.empty(len(rows), dtype=int)
    for row_idx, row in enumerate(rows):
        for col_idx, feature in enumerate(DIABETES_FEATURES):
            value = float(row[feature])
            if feature in DIABETES_ZERO_AS_MISSING and np.isclose(value, 0.0):
                value = np.nan
            X[row_idx, col_idx] = value
        y[row_idx] = int(row["Outcome"])

    return X, y


def _sample_rows(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic row sample without replacement."""
    if X.shape[0] <= n_samples:
        return X, y

    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], size=n_samples, replace=False)
    return X[indices], y[indices]


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(classification_report(y_true, y_pred)["accuracy"])


def _fit_score(model: object, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]) -> float:
    model.fit(X_train, y_train)
    return metric(y_test, model.predict(X_test))


def _impute_scale_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit preprocessing on train data and apply it to both splits."""
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train).astype(float)
    X_test = imputer.transform(X_test).astype(float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def _print_rows(title: str, rows: list[tuple[str, float, float]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'Model':<28} {'GlassBox':>10} {'Sklearn':>10} {'Ratio':>8}")
    for name, glassbox_score, sklearn_score in rows:
        ratio = glassbox_score / sklearn_score if sklearn_score != 0 else float("nan")
        print(f"{name:<28} {glassbox_score:>10.4f} {sklearn_score:>10.4f} {ratio:>8.2%}")


def main() -> None:
    (
        SkRandomForestClassifier,
        SkRandomForestRegressor,
        SkLinearRegression,
        SkLogisticRegression,
        accuracy_score,
        sk_r2_score,
        train_test_split,
        SkGaussianNB,
        KNeighborsClassifier,
        KNeighborsRegressor,
        fetch_california_housing,
        SkDecisionTreeClassifier,
        SkDecisionTreeRegressor,
    ) = _require_sklearn()

    X_cls, y_cls = _classification_data()
    X_reg, y_reg = _regression_data()
    X_diabetes, y_diabetes = _diabetes_data()
    housing = fetch_california_housing()
    X_housing, y_housing = _sample_rows(
        housing.data.astype(float),
        housing.target.astype(float),
        n_samples=CALIFORNIA_HOUSING_SAMPLE_SIZE,
        random_state=42,
    )

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
    )
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        X_diabetes, y_diabetes, test_size=0.3, random_state=42, stratify=y_diabetes
    )
    Xh_train, Xh_test, yh_train, yh_test = train_test_split(
        X_housing, y_housing, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    Xc_train = scaler.fit_transform(Xc_train)
    Xc_test = scaler.transform(Xc_test)

    scaler = StandardScaler()
    Xr_train = scaler.fit_transform(Xr_train)
    Xr_test = scaler.transform(Xr_test)
    Xd_train, Xd_test = _impute_scale_train_test(Xd_train, Xd_test)
    scaler = StandardScaler()
    Xh_train = scaler.fit_transform(Xh_train)
    Xh_test = scaler.transform(Xh_test)

    classification_rows = [
        (
            "LogisticRegression",
            _fit_score(LogisticRegression(learning_rate=0.1, n_iterations=1200), Xc_train, Xc_test, yc_train, yc_test, _accuracy),
            _fit_score(SkLogisticRegression(max_iter=1000), Xc_train, Xc_test, yc_train, yc_test, accuracy_score),
        ),
        (
            "DecisionTreeClassifier",
            _fit_score(DecisionTreeClassifier(max_depth=5), Xc_train, Xc_test, yc_train, yc_test, _accuracy),
            _fit_score(SkDecisionTreeClassifier(max_depth=5, random_state=42), Xc_train, Xc_test, yc_train, yc_test, accuracy_score),
        ),
        (
            "RandomForestClassifier",
            _fit_score(RandomForestClassifier(n_estimators=25, max_depth=6, random_state=42), Xc_train, Xc_test, yc_train, yc_test, _accuracy),
            _fit_score(SkRandomForestClassifier(n_estimators=25, max_depth=6, random_state=42), Xc_train, Xc_test, yc_train, yc_test, accuracy_score),
        ),
        (
            "GaussianNaiveBayes",
            _fit_score(GaussianNaiveBayes(), Xc_train, Xc_test, yc_train, yc_test, _accuracy),
            _fit_score(SkGaussianNB(), Xc_train, Xc_test, yc_train, yc_test, accuracy_score),
        ),
        (
            "KNearestNeighbors",
            _fit_score(KNearestNeighbors(k=5, task="classification"), Xc_train, Xc_test, yc_train, yc_test, _accuracy),
            _fit_score(KNeighborsClassifier(n_neighbors=5), Xc_train, Xc_test, yc_train, yc_test, accuracy_score),
        ),
    ]

    regression_rows = [
        (
            "LinearRegression",
            _fit_score(LinearRegression(learning_rate=0.05, n_iterations=1500), Xr_train, Xr_test, yr_train, yr_test, r2_score),
            _fit_score(SkLinearRegression(), Xr_train, Xr_test, yr_train, yr_test, sk_r2_score),
        ),
        (
            "DecisionTreeRegressor",
            _fit_score(DecisionTreeRegressor(max_depth=5), Xr_train, Xr_test, yr_train, yr_test, r2_score),
            _fit_score(SkDecisionTreeRegressor(max_depth=5, random_state=42), Xr_train, Xr_test, yr_train, yr_test, sk_r2_score),
        ),
        (
            "RandomForestRegressor",
            _fit_score(RandomForestRegressor(n_estimators=25, max_depth=6, random_state=42), Xr_train, Xr_test, yr_train, yr_test, r2_score),
            _fit_score(SkRandomForestRegressor(n_estimators=25, max_depth=6, random_state=42), Xr_train, Xr_test, yr_train, yr_test, sk_r2_score),
        ),
        (
            "KNearestNeighbors",
            _fit_score(KNearestNeighbors(k=5, task="regression"), Xr_train, Xr_test, yr_train, yr_test, r2_score),
            _fit_score(KNeighborsRegressor(n_neighbors=5), Xr_train, Xr_test, yr_train, yr_test, sk_r2_score),
        ),
    ]

    diabetes_rows = [
        (
            "LogisticRegression",
            _fit_score(LogisticRegression(learning_rate=0.1, n_iterations=1200), Xd_train, Xd_test, yd_train, yd_test, _accuracy),
            _fit_score(SkLogisticRegression(max_iter=1000), Xd_train, Xd_test, yd_train, yd_test, accuracy_score),
        ),
        (
            "DecisionTreeClassifier",
            _fit_score(DecisionTreeClassifier(max_depth=5), Xd_train, Xd_test, yd_train, yd_test, _accuracy),
            _fit_score(SkDecisionTreeClassifier(max_depth=5, random_state=42), Xd_train, Xd_test, yd_train, yd_test, accuracy_score),
        ),
        (
            "RandomForestClassifier",
            _fit_score(RandomForestClassifier(n_estimators=25, max_depth=6, random_state=42), Xd_train, Xd_test, yd_train, yd_test, _accuracy),
            _fit_score(SkRandomForestClassifier(n_estimators=25, max_depth=6, random_state=42), Xd_train, Xd_test, yd_train, yd_test, accuracy_score),
        ),
        (
            "GaussianNaiveBayes",
            _fit_score(GaussianNaiveBayes(), Xd_train, Xd_test, yd_train, yd_test, _accuracy),
            _fit_score(SkGaussianNB(), Xd_train, Xd_test, yd_train, yd_test, accuracy_score),
        ),
        (
            "KNearestNeighbors",
            _fit_score(KNearestNeighbors(k=5, task="classification"), Xd_train, Xd_test, yd_train, yd_test, _accuracy),
            _fit_score(KNeighborsClassifier(n_neighbors=5), Xd_train, Xd_test, yd_train, yd_test, accuracy_score),
        ),
    ]

    online_regression_rows = [
        (
            "LinearRegression",
            _fit_score(LinearRegression(learning_rate=0.05, n_iterations=1500), Xh_train, Xh_test, yh_train, yh_test, r2_score),
            _fit_score(SkLinearRegression(), Xh_train, Xh_test, yh_train, yh_test, sk_r2_score),
        ),
        (
            "DecisionTreeRegressor",
            _fit_score(DecisionTreeRegressor(max_depth=6), Xh_train, Xh_test, yh_train, yh_test, r2_score),
            _fit_score(SkDecisionTreeRegressor(max_depth=6, random_state=42), Xh_train, Xh_test, yh_train, yh_test, sk_r2_score),
        ),
        (
            "RandomForestRegressor",
            _fit_score(RandomForestRegressor(n_estimators=10, max_depth=6, random_state=42), Xh_train, Xh_test, yh_train, yh_test, r2_score),
            _fit_score(SkRandomForestRegressor(n_estimators=10, max_depth=6, random_state=42), Xh_train, Xh_test, yh_train, yh_test, sk_r2_score),
        ),
        (
            "KNearestNeighbors",
            _fit_score(KNearestNeighbors(k=5, task="regression"), Xh_train, Xh_test, yh_train, yh_test, r2_score),
            _fit_score(KNeighborsRegressor(n_neighbors=5), Xh_train, Xh_test, yh_train, yh_test, sk_r2_score),
        ),
    ]

    _print_rows("Classification Accuracy", classification_rows)
    _print_rows("Regression R2", regression_rows)
    _print_rows("Diabetes CSV Accuracy", diabetes_rows)
    _print_rows(
        f"California Housing Online Regression ({X_housing.shape[0]} rows)",
        online_regression_rows,
    )


if __name__ == "__main__":
    main()
