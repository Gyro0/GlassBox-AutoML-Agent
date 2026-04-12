"""Reporting utilities for AutoFit results."""

from __future__ import annotations

from typing import Any

import numpy as np


def make_json_safe(value: Any) -> Any:
    """Recursively convert Python and NumPy values into JSON-safe objects."""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        return numeric_value if np.isfinite(numeric_value) else None

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    if value is None or isinstance(value, str):
        return value

    return str(value)


def _extract_feature_importances(best_result: dict[str, Any]) -> dict[str, float]:
    """Build a feature-importance-style mapping from the best estimator."""
    estimator = best_result.get("best_estimator")
    feature_names = best_result.get("feature_names") or []
    if estimator is None or not feature_names:
        return {}

    if hasattr(estimator, "feature_importances_"):
        raw_values = np.asarray(getattr(estimator, "feature_importances_"), dtype=float)
    elif hasattr(estimator, "weights_"):
        weights = np.asarray(getattr(estimator, "weights_"), dtype=float)
        if weights.shape[0] == len(feature_names) + 1:
            raw_values = weights[1:]
        else:
            raw_values = weights[: len(feature_names)]
    else:
        return {}

    feature_importances: dict[str, float] = {}
    for index, name in enumerate(feature_names):
        if index >= raw_values.shape[0]:
            break
        feature_importances[name] = float(raw_values[index])
    return feature_importances


def generate_report(
    eda_summary: dict[str, Any],
    best_result: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured AutoFit report payload."""
    best_model = best_result.get("best_model")
    if not isinstance(best_model, str):
        estimator = best_result.get("best_estimator")
        best_model = estimator.__class__.__name__ if estimator is not None else None

    report = {
        "eda_summary": eda_summary,
        "best_model": best_model,
        "best_params": dict(best_result.get("best_params") or {}),
        "cv_score": float(best_result.get("cv_score") or 0.0),
        "feature_importances": _extract_feature_importances(best_result),
    }
    return make_json_safe(report)
