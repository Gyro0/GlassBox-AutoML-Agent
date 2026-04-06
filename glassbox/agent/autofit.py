"""AutoFit - End-to-end AutoML pipeline."""

import numpy as np


def auto_fit(
    csv_path: str,
    target_column: str,
    task: str = "auto",
    search: str = "random",
    time_budget: int = 120):
    """
    Runs full AutoML pipeline: EDA -> Preprocessing -> Model Search -> Best Model.

    Args:
        csv_path: Path to CSV file
        target_column: Name of target column
        task: "classification", "regression", or "auto" (infer from target)
        search: "grid" or "random"
        time_budget: Max seconds for hyperparameter search

    Returns:
        dict with best_model, best_params, cv_score, eda_summary
    """
    return {
        "best_model": None,
        "best_params": {},
        "cv_score": 0.0,
        "eda_summary": {}
        }