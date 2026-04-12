"""AutoFit - End-to-end AutoML pipeline."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from glassbox.agent.report import generate_report, make_json_safe
from glassbox.eda import (
    build_pearson_correlation_matrix,
    infer_column_types,
    iqr_outlier_handler,
    profile_numeric_columns,
)
from glassbox.evaluation.classification import classification_report
from glassbox.evaluation.regression import r2_score
from glassbox.optimization import GridSearchCV, KFoldCV, RandomSearchCV
from glassbox.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler

_MISSING_TOKENS = {"", "na", "nan", "none", "null"}


@dataclass(frozen=True)
class ModelSpec:
    """Declarative AutoFit model configuration."""

    name: str
    module_path: str
    class_name: str
    search_space: dict[str, list[Any]]
    scoring_fn: Callable[[np.ndarray, np.ndarray], float]


_MODEL_REGISTRY: dict[str, list[ModelSpec]] = {
    "regression": [
        ModelSpec(
            name="LinearRegression",
            module_path="glassbox.models.linear",
            class_name="LinearRegression",
            search_space={
                "learning_rate": [0.01, 0.05, 0.1],
                "n_iterations": [300, 800, 1500],
                "tol": [1e-7, 1e-6],
            },
            scoring_fn=r2_score,
        ),
    ],
    "classification": [
        ModelSpec(
            name="LogisticRegression",
            module_path="glassbox.models.linear",
            class_name="LogisticRegression",
            search_space={
                "learning_rate": [0.01, 0.05, 0.1],
                "n_iterations": [300, 800, 1500],
                "tol": [1e-7, 1e-6],
                "threshold": [0.4, 0.5, 0.6],
            },
            scoring_fn=lambda y_true, y_pred: float(
                classification_report(y_true, y_pred)["accuracy"]
            ),
        ),
    ],
}


def _is_missing(value: Any) -> bool:
    """Return True when a CSV value should be treated as missing."""
    if value is None:
        return True
    return str(value).strip().lower() in _MISSING_TOKENS


def _clean_text(value: Any) -> str:
    """Normalize text values loaded from CSV."""
    return str(value).strip()


def _can_parse_float(values: list[str]) -> bool:
    """Return True when every non-missing value can be parsed as a float."""
    if not values:
        return False

    for value in values:
        try:
            float(value)
        except ValueError:
            return False
    return True


def _read_csv_columns(csv_path: str) -> tuple[list[str], dict[str, list[str]]]:
    """Load a CSV file into a column-oriented dictionary."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError("CSV file must include a header row.")

        columns = {field: [] for field in fieldnames}
        for row in reader:
            for field in fieldnames:
                columns[field].append(row.get(field, ""))

    if not columns or len(next(iter(columns.values()), [])) == 0:
        raise ValueError("CSV file must contain at least one data row.")

    return fieldnames, columns


def _prepare_columns_for_type_inference(
    raw_columns: dict[str, list[str]],
) -> dict[str, np.ndarray]:
    """Convert columns to arrays that work well with AutoTyper."""
    prepared: dict[str, np.ndarray] = {}

    for name, values in raw_columns.items():
        observed = [_clean_text(value) for value in values if not _is_missing(value)]
        if not observed:
            prepared[name] = np.array([], dtype=object)
            continue

        if _can_parse_float(observed):
            prepared[name] = np.array(
                [np.nan if _is_missing(value) else float(_clean_text(value)) for value in values],
                dtype=float,
            )
        else:
            prepared[name] = np.array(
                ["" if _is_missing(value) else _clean_text(value) for value in values],
                dtype=object,
            )

    return prepared


def _coerce_numeric_column(values: list[str], name: str) -> np.ndarray:
    """Parse a feature column into a float array with NaN for missing values."""
    coerced: list[float] = []
    for value in values:
        if _is_missing(value):
            coerced.append(np.nan)
            continue

        try:
            coerced.append(float(_clean_text(value)))
        except ValueError as exc:
            raise ValueError(f"Column '{name}' contains a non-numeric value: {value!r}") from exc

    return np.asarray(coerced, dtype=float)


def _mode_text(values: list[str], column_name: str) -> str:
    """Return the most frequent non-missing categorical value."""
    observed = [_clean_text(value) for value in values if not _is_missing(value)]
    if not observed:
        raise ValueError(f"Cannot impute column '{column_name}': entire column is missing.")

    counts = Counter(observed)
    max_count = max(counts.values())
    candidates = sorted(value for value, count in counts.items() if count == max_count)
    return candidates[0]


def _impute_categorical_columns(
    raw_columns: dict[str, list[str]],
    column_names: list[str],
) -> np.ndarray:
    """Fill missing categorical values with each column's mode."""
    if not column_names:
        n_rows = len(next(iter(raw_columns.values()))) if raw_columns else 0
        return np.empty((n_rows, 0), dtype=object)

    filled_columns: list[np.ndarray] = []
    for name in column_names:
        fill_value = _mode_text(raw_columns[name], name)
        filled = [
            fill_value if _is_missing(value) else _clean_text(value)
            for value in raw_columns[name]
        ]
        filled_columns.append(np.asarray(filled, dtype=object))

    return np.column_stack(filled_columns)


def _cap_outliers_per_column(
    numeric_data: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]]]:
    """Cap outliers on each numeric column while preserving NaN positions."""
    capped_columns: dict[str, np.ndarray] = {}
    outlier_rows: dict[str, list[int]] = {}

    for name, values in numeric_data.items():
        capped = values.copy()
        non_missing_mask = ~np.isnan(values)
        non_missing_values = values[non_missing_mask]

        if non_missing_values.size == 0:
            outlier_rows[name] = []
            capped_columns[name] = capped
            continue

        local_rows = iqr_outlier_handler(
            non_missing_values.reshape(-1, 1),
            mode="flag",
            return_row_indices=True,
        )
        outlier_rows[name] = np.flatnonzero(non_missing_mask)[local_rows].tolist()

        capped_non_missing = iqr_outlier_handler(
            non_missing_values.reshape(-1, 1),
            mode="cap",
        ).reshape(-1)
        capped[non_missing_mask] = capped_non_missing
        capped_columns[name] = capped

    return capped_columns, outlier_rows


def _build_numerical_summary(
    numeric_matrix: np.ndarray,
    numeric_feature_names: list[str],
) -> dict[str, Any]:
    """Build EDA outputs for numerical features."""
    if numeric_matrix.shape[1] == 0:
        return {
            "numerical_profile": {},
            "correlation_matrix": [],
            "correlation_feature_names": [],
            "high_collinearity": [],
        }

    profile = profile_numeric_columns(numeric_matrix, numeric_feature_names)
    corr_matrix, corr_names, high_collinearity = build_pearson_correlation_matrix(
        numeric_matrix,
        feature_names=numeric_feature_names,
    )
    return {
        "numerical_profile": profile,
        "correlation_matrix": corr_matrix.tolist(),
        "correlation_feature_names": corr_names,
        "high_collinearity": [
            {"feature_a": first, "feature_b": second, "correlation": float(value)}
            for first, second, value in high_collinearity
        ],
    }


def _build_one_hot_feature_names(
    input_feature_names: list[str],
    categories: list[np.ndarray],
) -> list[str]:
    """Derive output feature names from OneHotEncoder categories."""
    encoded_names: list[str] = []
    for feature_name, category_values in zip(input_feature_names, categories):
        for category in category_values.tolist():
            encoded_names.append(f"{feature_name}={category}")
    return encoded_names


def _resolve_task(task: str, target_type: str) -> str:
    """Resolve task type from user input and target metadata."""
    if task not in {"auto", "classification", "regression"}:
        raise ValueError("task must be 'auto', 'classification', or 'regression'.")

    if task != "auto":
        return task
    return "regression" if target_type == "numerical" else "classification"


def _prepare_target(values: list[str], task: str, column_name: str) -> np.ndarray:
    """Convert the target column into a model-ready NumPy array."""
    if any(_is_missing(value) for value in values):
        raise ValueError(f"Target column '{column_name}' cannot contain missing values.")

    cleaned = [_clean_text(value) for value in values]
    if task == "regression":
        try:
            return np.asarray([float(value) for value in cleaned], dtype=float)
        except ValueError as exc:
            raise ValueError(
                f"Regression target column '{column_name}' must be numeric."
            ) from exc

    if _can_parse_float(cleaned):
        numeric_target = np.asarray([float(value) for value in cleaned], dtype=float)
        if np.all(np.isclose(numeric_target, np.round(numeric_target))):
            return numeric_target.astype(int)
        return numeric_target

    return np.asarray(cleaned, dtype=object)


def _load_model_class(spec: ModelSpec) -> type | None:
    """Best-effort import for a model spec."""
    try:
        module = import_module(spec.module_path)
    except ImportError:
        return None

    model_class = getattr(module, spec.class_name, None)
    if isinstance(model_class, type):
        return model_class
    return None


def _get_candidate_models(task: str) -> list[tuple[ModelSpec, type]]:
    """Resolve the currently available models for a task."""
    available_models: list[tuple[ModelSpec, type]] = []
    for spec in _MODEL_REGISTRY[task]:
        model_class = _load_model_class(spec)
        if model_class is not None:
            available_models.append((spec, model_class))

    if not available_models:
        raise RuntimeError(f"No available models are registered for task '{task}'.")

    return available_models


def _build_searcher(
    search: str,
    model_class: type,
    search_space: dict[str, list[Any]],
    cv: KFoldCV,
    scoring_fn: Callable[[np.ndarray, np.ndarray], float],
    time_budget: int,
) -> GridSearchCV | RandomSearchCV:
    """Create a hyperparameter searcher for one model family."""
    if search == "grid":
        return GridSearchCV(model_class, search_space, cv, scoring_fn)

    return RandomSearchCV(
        model_class=model_class,
        param_distributions=search_space,
        n_iter=6,
        time_budget_seconds=time_budget,
        cv=cv,
        scoring_fn=scoring_fn,
    )


def _serialize_search_results(
    model_name: str,
    raw_results: list[tuple[float, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Convert internal search results to a JSON-safe summary."""
    return [
        {
            "model": model_name,
            "score": float(score),
            "params": dict(params),
        }
        for score, params in raw_results
    ]


class AutoFit:
    """Small AutoML runner built around the currently available modules."""

    def __init__(
        self,
        task: str = "auto",
        search: str = "random",
        time_budget: int = 120,
    ) -> None:
        if search not in {"grid", "random"}:
            raise ValueError("search must be either 'grid' or 'random'.")

        self.task = task
        self.search = search
        self.time_budget = time_budget
        self.result_: dict[str, Any] | None = None
        self.best_estimator_: object | None = None
        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float | None = None
        self.search_results_: list[dict[str, Any]] = []

    def fit(self, csv_path: str, target_column: str) -> "AutoFit":
        """Run the current AutoFit pipeline and store the result."""
        fieldnames, raw_columns = _read_csv_columns(csv_path)
        if target_column not in raw_columns:
            raise ValueError(f"Target column '{target_column}' was not found in the CSV header.")

        typed_columns = _prepare_columns_for_type_inference(raw_columns)
        inferred_types = infer_column_types(typed_columns)
        resolved_task = _resolve_task(self.task, inferred_types[target_column])

        feature_names = [name for name in fieldnames if name != target_column]
        if not feature_names:
            raise ValueError("AutoFit requires at least one feature column.")

        numeric_feature_names = [
            name for name in feature_names if inferred_types[name] == "numerical"
        ]
        categorical_feature_names = [
            name for name in feature_names if inferred_types[name] in {"categorical", "boolean"}
        ]

        numeric_columns = {
            name: _coerce_numeric_column(raw_columns[name], name)
            for name in numeric_feature_names
        }
        capped_numeric_columns, outlier_rows = _cap_outliers_per_column(numeric_columns)

        if numeric_feature_names:
            numeric_matrix = np.column_stack(
                [capped_numeric_columns[name] for name in numeric_feature_names]
            )
            numeric_imputer = SimpleImputer(strategy="mean")
            numeric_imputed = numeric_imputer.fit_transform(numeric_matrix)
            numeric_summary = _build_numerical_summary(
                numeric_imputed,
                numeric_feature_names,
            )
            scaler = StandardScaler()
            numeric_processed = scaler.fit_transform(numeric_imputed)
        else:
            n_rows = len(raw_columns[target_column])
            numeric_summary = _build_numerical_summary(
                np.empty((n_rows, 0), dtype=float),
                [],
            )
            numeric_processed = np.empty((n_rows, 0), dtype=float)

        categorical_matrix = _impute_categorical_columns(raw_columns, categorical_feature_names)
        categorical_feature_output_names: list[str] = []
        if categorical_feature_names:
            encoder = OneHotEncoder()
            categorical_processed = encoder.fit_transform(categorical_matrix).astype(float)
            categorical_feature_output_names = _build_one_hot_feature_names(
                categorical_feature_names,
                encoder.categories_,
            )
        else:
            categorical_processed = np.empty((len(raw_columns[target_column]), 0), dtype=float)

        if numeric_processed.shape[1] == 0 and categorical_processed.shape[1] == 0:
            raise ValueError("AutoFit requires at least one usable feature column.")

        feature_matrix_parts = [
            part for part in (numeric_processed, categorical_processed) if part.shape[1] > 0
        ]
        X_processed = np.hstack(feature_matrix_parts).astype(float)
        processed_feature_names = numeric_feature_names + categorical_feature_output_names

        y = _prepare_target(raw_columns[target_column], resolved_task, target_column)
        if y.shape[0] < 2:
            raise ValueError("AutoFit requires at least two samples.")

        cv_splits = min(5, y.shape[0])
        if cv_splits < 2:
            raise ValueError("AutoFit requires at least two samples for cross-validation.")
        cv = KFoldCV(n_splits=cv_splits, shuffle=True, random_state=42)

        candidate_models = _get_candidate_models(resolved_task)
        all_search_results: list[dict[str, Any]] = []
        best_searcher: GridSearchCV | RandomSearchCV | None = None
        best_model_name: str | None = None
        best_score = -np.inf

        for spec, model_class in candidate_models:
            searcher = _build_searcher(
                search=self.search,
                model_class=model_class,
                search_space=spec.search_space,
                cv=cv,
                scoring_fn=spec.scoring_fn,
                time_budget=self.time_budget,
            )
            searcher.fit(X_processed, y)
            all_search_results.extend(_serialize_search_results(spec.name, searcher.results_))

            searcher_score = float(searcher.best_score_ if searcher.best_score_ is not None else -np.inf)
            if best_searcher is None or searcher_score > best_score:
                best_searcher = searcher
                best_model_name = spec.name
                best_score = searcher_score

        all_search_results.sort(key=lambda item: item["score"], reverse=True)

        if best_searcher is None or best_searcher.best_model_ is None:
            raise RuntimeError("AutoFit could not select a best model.")

        eda_summary = {
            "column_types": {name: inferred_types[name] for name in feature_names},
            "target_type": inferred_types[target_column],
            "outlier_rows": outlier_rows,
            **numeric_summary,
        }

        best_result = {
            "best_model": best_model_name or best_searcher.best_model_.__class__.__name__,
            "best_params": best_searcher.best_params_,
            "cv_score": best_searcher.best_score_,
            "best_estimator": best_searcher.best_model_,
            "feature_names": processed_feature_names,
        }
        result = generate_report(eda_summary, best_result)
        result.update(
            {
                "task": resolved_task,
                "search": self.search,
                "feature_names": processed_feature_names,
                "search_results": all_search_results,
                "processed_shape": list(X_processed.shape),
                "candidate_models": [spec.name for spec, _ in candidate_models],
            }
        )

        self.result_ = make_json_safe(result)
        self.best_estimator_ = best_searcher.best_model_
        self.best_params_ = best_searcher.best_params_
        self.best_score_ = best_searcher.best_score_
        self.search_results_ = all_search_results
        return self

    def run(self, csv_path: str, target_column: str) -> dict[str, Any]:
        """Convenience wrapper that returns the pipeline result directly."""
        return self.fit(csv_path, target_column).result_


def auto_fit(
    csv_path: str,
    target_column: str,
    task: str = "auto",
    search: str = "random",
    time_budget: int = 120,
) -> dict[str, Any]:
    """Run the current AutoFit pipeline and return a structured result."""
    runner = AutoFit(task=task, search=search, time_budget=time_budget)
    return runner.run(csv_path, target_column)
