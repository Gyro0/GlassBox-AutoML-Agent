"""Tests for the agent layer."""

from __future__ import annotations

import csv
import json

import numpy as np

from glassbox.agent import AutoFit, auto_fit, generate_report


def _write_csv(tmp_path, filename: str, headers: list[str], rows: list[list[object]]) -> str:
    """Write a small CSV fixture to disk and return its path."""
    csv_path = tmp_path / filename
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return str(csv_path)


def test_agent_exports_public_interfaces() -> None:
    assert AutoFit.__name__ == "AutoFit"
    assert callable(auto_fit)
    assert callable(generate_report)


def test_autofit_regression_pipeline_returns_structured_result(tmp_path) -> None:
    csv_path = _write_csv(
        tmp_path,
        "regression.csv",
        ["x1", "x2", "target"],
        [
            [1, 10, 6],
            [2, "", 8],
            [3, 30, 10],
            [4, 40, 12],
            [5, 50, 14],
            [6, 60, 16],
            [7, 70, 18],
            [8, 80, 20],
            [9, 90, 22],
            [10, 100, 24],
        ],
    )

    runner = AutoFit(task="regression", search="grid", time_budget=10)
    runner.fit(csv_path, "target")
    result = runner.result_

    assert runner.best_estimator_ is not None
    assert result is not None
    assert result["task"] == "regression"
    assert result["best_model"] == "LinearRegression"
    assert result["cv_score"] > 0.90
    assert result["feature_names"] == ["x1", "x2"]
    assert "x1" in result["eda_summary"]["column_types"]
    assert "x1" in result["feature_importances"]
    assert "best_estimator" not in result
    assert result["candidate_models"] == ["LinearRegression"]
    assert json.loads(json.dumps(result))["best_model"] == "LinearRegression"


def test_autofit_auto_detects_classification_and_handles_categorical_data(tmp_path) -> None:
    csv_path = _write_csv(
        tmp_path,
        "classification.csv",
        ["age", "country", "device", "target"],
        [
            [21, "US", "mobile", "yes"],
            [22, "CA", "desktop", "no"],
            [23, "US", "", "yes"],
            [24, "CA", "desktop", "no"],
            [25, "US", "mobile", "yes"],
            [26, "CA", "tablet", "no"],
            [27, "US", "desktop", "yes"],
            [28, "CA", "tablet", "no"],
            [29, "US", "mobile", "yes"],
            [30, "CA", "desktop", "no"],
        ],
    )

    result = auto_fit(csv_path, "target", task="auto", search="grid", time_budget=10)

    assert result["task"] == "classification"
    assert result["best_model"] == "LogisticRegression"
    assert result["cv_score"] >= 0.80
    assert any(name.startswith("country=") for name in result["feature_names"])
    assert result["eda_summary"]["column_types"]["country"] == "categorical"
    assert result["candidate_models"] == ["LogisticRegression"]
    assert result["search_results"][0]["model"] == "LogisticRegression"
    assert json.loads(json.dumps(result))["task"] == "classification"


def test_generate_report_returns_json_safe_payload() -> None:
    class DummyEstimator:
        weights_ = np.array([0.0, 1.5, -2.0], dtype=np.float64)

    report = generate_report(
        eda_summary={"matrix": np.array([[1.0, 2.0], [3.0, 4.0]])},
        best_result={
            "best_model": "LinearRegression",
            "best_params": {"learning_rate": np.float64(0.1)},
            "cv_score": np.float64(0.95),
            "best_estimator": DummyEstimator(),
            "feature_names": ["f1", "f2"],
        },
    )

    assert report["cv_score"] == 0.95
    assert report["best_params"]["learning_rate"] == 0.1
    assert report["feature_importances"] == {"f1": 1.5, "f2": -2.0}
    assert json.loads(json.dumps(report))["best_model"] == "LinearRegression"
