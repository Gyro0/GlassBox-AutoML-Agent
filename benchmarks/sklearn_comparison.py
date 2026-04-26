"""Compare GlassBox models with equivalent Scikit-Learn baselines on a CSV.

Run from the repository root:

    python benchmarks/sklearn_comparison.py --task regression --csv data/_uploaded.csv --target Delay
    python benchmarks/sklearn_comparison.py --task classification --csv data/classification.csv --target stroke

Scikit-Learn is intentionally used only in this benchmark script, never in
the ``glassbox`` package itself.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from glassbox.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler

MISSING_TOKENS = {"", "na", "nan", "none", "null"}


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
        from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
        from sklearn.tree import DecisionTreeRegressor as SkDecisionTreeRegressor
    except ImportError as exc:
        raise SystemExit(
            "Scikit-Learn is required for this benchmark. Install it with "
            "`python -m pip install scikit-learn`."
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
        SkDecisionTreeClassifier,
        SkDecisionTreeRegressor,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GlassBox vs sklearn models on a CSV dataset.",
    )
    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Benchmark task type.",
    )
    parser.add_argument(
        "--csv",
        default="data/_uploaded.csv",
        help="Path to CSV file (absolute or relative to repository root).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target column name. Defaults to the last column when omitted.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test split fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    return parser.parse_args()


def _resolve_csv_path(path_text: str) -> Path:
    csv_path = Path(path_text).expanduser()
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()
    return csv_path


def _resolve_target_column(fieldnames: list[str], requested_target: str | None) -> str:
    if requested_target is None or not _clean_text(requested_target):
        return fieldnames[-1]
    target = _clean_text(requested_target)
    if target not in fieldnames:
        raise ValueError(f"Target column '{target}' not found in CSV header.")
    return target


def _is_missing(value: object) -> bool:
    return value is None or str(value).strip().lower() in MISSING_TOKENS


def _clean_text(value: object) -> str:
    return str(value).strip()


def _can_parse_float(values: list[str]) -> bool:
    if not values:
        return False
    for value in values:
        try:
            float(value)
        except ValueError:
            return False
    return True


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [_clean_text(name) for name in (reader.fieldnames or []) if name is not None]
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            cleaned_row = {
                _clean_text(key): ("" if value is None else value)
                for key, value in raw_row.items()
                if key is not None
            }
            if not cleaned_row:
                continue
            rows.append(cleaned_row)

    if not rows or not fieldnames:
        raise ValueError("CSV is empty or missing a valid header row.")

    return fieldnames, rows


def _prepare_schema(
    fieldnames: list[str],
    rows: list[dict[str, str]],
    target_column: str,
    task: str,
) -> tuple[list[dict[str, str]], list[str], list[str], list[str], int, list[str]]:
    if target_column not in fieldnames:
        raise ValueError(f"Target column '{target_column}' not found in CSV header.")

    feature_names = [name for name in fieldnames if name != target_column]
    if not feature_names:
        raise ValueError("CSV needs at least one feature column and one target column.")

    valid_rows: list[dict[str, str]] = []
    dropped_target_rows = 0
    for row in rows:
        raw_target = row.get(target_column)
        if _is_missing(raw_target):
            dropped_target_rows += 1
            continue

        if task == "regression":
            try:
                float(_clean_text(raw_target))
            except ValueError:
                dropped_target_rows += 1
                continue

        valid_rows.append(row)

    if not valid_rows:
        if task == "regression":
            raise ValueError("No valid target values remain after filtering missing/non-numeric target rows.")
        raise ValueError("No valid target values remain after filtering missing target rows.")

    target_levels: list[str] = []
    if task == "classification":
        target_levels = sorted({_clean_text(row[target_column]) for row in valid_rows})
        if len(target_levels) < 2:
            raise ValueError("Classification task requires at least two target classes.")

    numeric_features: list[str] = []
    categorical_features: list[str] = []
    dropped_all_missing_features: list[str] = []

    for name in feature_names:
        observed = [
            _clean_text(row.get(name))
            for row in valid_rows
            if not _is_missing(row.get(name))
        ]
        if not observed:
            dropped_all_missing_features.append(name)
            continue

        if _can_parse_float(observed):
            numeric_features.append(name)
        else:
            categorical_features.append(name)

    if not numeric_features and not categorical_features:
        raise ValueError("No usable feature columns found after filtering all-missing columns.")

    return (
        valid_rows,
        numeric_features,
        categorical_features,
        dropped_all_missing_features,
        dropped_target_rows,
        target_levels,
    )


def _rows_to_feature_blocks(
    rows: list[dict[str, str]],
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(rows)

    if numeric_features:
        numeric = np.asarray(
            [
                [
                    np.nan if _is_missing(row.get(name)) else float(_clean_text(row.get(name)))
                    for name in numeric_features
                ]
                for row in rows
            ],
            dtype=float,
        )
    else:
        numeric = np.empty((n_rows, 0), dtype=float)

    if categorical_features:
        categorical = np.asarray(
            [
                ["" if _is_missing(row.get(name)) else _clean_text(row.get(name)) for name in categorical_features]
                for row in rows
            ],
            dtype=object,
        )
    else:
        categorical = np.empty((n_rows, 0), dtype=object)

    return numeric, categorical


def _build_design_matrices(
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_numeric, train_categorical = _rows_to_feature_blocks(
        train_rows,
        numeric_features,
        categorical_features,
    )
    test_numeric, test_categorical = _rows_to_feature_blocks(
        test_rows,
        numeric_features,
        categorical_features,
    )

    if train_numeric.shape[1] > 0:
        numeric_imputer = SimpleImputer(strategy="mean")
        train_numeric = numeric_imputer.fit_transform(train_numeric).astype(float)
        test_numeric = numeric_imputer.transform(test_numeric).astype(float)

        scaler = StandardScaler()
        train_numeric = scaler.fit_transform(train_numeric)
        test_numeric = scaler.transform(test_numeric)

    if train_categorical.shape[1] > 0:
        categorical_imputer = SimpleImputer(strategy="mode")
        train_categorical = categorical_imputer.fit_transform(train_categorical)
        test_categorical = categorical_imputer.transform(test_categorical)

        encoder = OneHotEncoder()
        train_categorical = encoder.fit_transform(train_categorical).astype(float)
        test_categorical = encoder.transform(test_categorical).astype(float)

    train_parts = [part for part in (train_numeric, train_categorical) if part.shape[1] > 0]
    test_parts = [part for part in (test_numeric, test_categorical) if part.shape[1] > 0]

    X_train = np.hstack(train_parts).astype(float)
    X_test = np.hstack(test_parts).astype(float)
    return X_train, X_test


def _target_array(rows: list[dict[str, str]], target_column: str, task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray(
            [float(_clean_text(row[target_column])) for row in rows],
            dtype=float,
        )
    return np.asarray(
        [_clean_text(row[target_column]) for row in rows],
        dtype=object,
    )


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _fit_score(model: object, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float]) -> float:
    model.fit(X_train, y_train)
    return metric(y_test, model.predict(X_test))


def _print_rows(title: str, rows: list[tuple[str, float, float]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'Model':<28} {'GlassBox':>10} {'Sklearn':>10} {'Ratio':>8}")
    for name, glassbox_score, sklearn_score in rows:
        ratio = glassbox_score / sklearn_score if sklearn_score != 0 else float("nan")
        print(f"{name:<28} {glassbox_score:>10.4f} {sklearn_score:>10.4f} {ratio:>8.2%}")


def main() -> None:
    args = _parse_args()
    csv_path = _resolve_csv_path(args.csv)

    fieldnames, rows = _read_csv_rows(csv_path)
    target_column = _resolve_target_column(fieldnames, args.target)
    (
        valid_rows,
        numeric_features,
        categorical_features,
        dropped_all_missing_features,
        dropped_target_rows,
        target_levels,
    ) = _prepare_schema(
        fieldnames=fieldnames,
        rows=rows,
        target_column=target_column,
        task=args.task,
    )

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
        SkDecisionTreeClassifier,
        SkDecisionTreeRegressor,
    ) = _require_sklearn()

    y_all = _target_array(valid_rows, target_column, args.task)
    split_kwargs: dict[str, object] = {
        "test_size": args.test_size,
        "random_state": args.random_state,
    }
    if args.task == "classification":
        split_kwargs["stratify"] = y_all

    train_rows, test_rows = train_test_split(
        valid_rows,
        **split_kwargs,
    )

    X_train, X_test = _build_design_matrices(
        train_rows,
        test_rows,
        numeric_features,
        categorical_features,
    )
    y_train = _target_array(train_rows, target_column, args.task)
    y_test = _target_array(test_rows, target_column, args.task)

    print(
        "\nCSV Classification Benchmark"
        if args.task == "classification"
        else "\nCSV Regression Benchmark"
    )
    print("---------------------------" if args.task == "classification" else "-----------------------")
    print(f"CSV path: {csv_path}")
    print(f"Task: {args.task}")
    print(f"Target: {target_column}")
    if args.task == "regression":
        print(
            "Rows used: "
            f"{len(valid_rows)} (dropped target-missing/non-numeric rows: {dropped_target_rows})"
        )
    else:
        print(
            "Rows used: "
            f"{len(valid_rows)} (dropped target-missing rows: {dropped_target_rows})"
        )
        print(f"Classes: {', '.join(target_levels)}")
    print(
        "Features: "
        f"{len(numeric_features)} numeric, {len(categorical_features)} categorical"
    )
    if dropped_all_missing_features:
        print(
            "Dropped all-missing feature columns: "
            + ", ".join(dropped_all_missing_features)
        )
    print(f"Train shape: {X_train.shape}; Test shape: {X_test.shape}")

    if args.task == "regression":
        regression_rows = [
            (
                "LinearRegression",
                _fit_score(
                    LinearRegression(learning_rate=0.05, n_iterations=1500),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    r2_score,
                ),
                _fit_score(
                    SkLinearRegression(),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    sk_r2_score,
                ),
            ),
            (
                "DecisionTreeRegressor",
                _fit_score(
                    DecisionTreeRegressor(max_depth=5),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    r2_score,
                ),
                _fit_score(
                    SkDecisionTreeRegressor(max_depth=5, random_state=args.random_state),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    sk_r2_score,
                ),
            ),
            (
                "RandomForestRegressor",
                _fit_score(
                    RandomForestRegressor(n_estimators=25, max_depth=6, random_state=args.random_state),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    r2_score,
                ),
                _fit_score(
                    SkRandomForestRegressor(n_estimators=25, max_depth=6, random_state=args.random_state),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    sk_r2_score,
                ),
            ),
            (
                "KNearestNeighbors",
                _fit_score(
                    KNearestNeighbors(k=5, task="regression"),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    r2_score,
                ),
                _fit_score(
                    KNeighborsRegressor(n_neighbors=5),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    sk_r2_score,
                ),
            ),
        ]
        _print_rows("Regression R2", regression_rows)
        return

    classification_rows: list[tuple[str, float, float]] = []
    unique_classes = np.unique(y_train)

    if unique_classes.shape[0] == 2:
        classification_rows.append(
            (
                "LogisticRegression",
                _fit_score(
                    LogisticRegression(learning_rate=0.1, n_iterations=1200),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    _accuracy,
                ),
                _fit_score(
                    SkLogisticRegression(max_iter=1000),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    accuracy_score,
                ),
            )
        )
    else:
        print("Skipping LogisticRegression: requires exactly two classes.")

    classification_rows.extend(
        [
            (
                "DecisionTreeClassifier",
                _fit_score(
                    DecisionTreeClassifier(max_depth=5),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    _accuracy,
                ),
                _fit_score(
                    SkDecisionTreeClassifier(max_depth=5, random_state=args.random_state),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    accuracy_score,
                ),
            ),
            (
                "RandomForestClassifier",
                _fit_score(
                    RandomForestClassifier(n_estimators=25, max_depth=6, random_state=args.random_state),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    _accuracy,
                ),
                _fit_score(
                    SkRandomForestClassifier(n_estimators=25, max_depth=6, random_state=args.random_state),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    accuracy_score,
                ),
            ),
            (
                "GaussianNaiveBayes",
                _fit_score(
                    GaussianNaiveBayes(),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    _accuracy,
                ),
                _fit_score(
                    SkGaussianNB(),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    accuracy_score,
                ),
            ),
            (
                "KNearestNeighbors",
                _fit_score(
                    KNearestNeighbors(k=5, task="classification"),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    _accuracy,
                ),
                _fit_score(
                    KNeighborsClassifier(n_neighbors=5),
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    accuracy_score,
                ),
            ),
        ]
    )

    _print_rows("Classification Accuracy", classification_rows)


if __name__ == "__main__":
    main()
