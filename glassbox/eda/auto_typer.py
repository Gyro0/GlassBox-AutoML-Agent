"""Automatic column type inference for EDA workflows."""

from __future__ import annotations

from typing import Any

import numpy as np


def infer_column_types(
    data: dict[str, list[Any] | np.ndarray],
    unique_ratio_threshold: float = 0.1,
) -> dict[str, str]:
    """Infer whether each column is numerical, categorical, or boolean."""
    inferred: dict[str, str] = {}

    for column_name, values in data.items():
        array = np.asarray(values)
        if array.size == 0:
            inferred[column_name] = "categorical"
            continue

        unique_values = np.unique(array)
        unique_count = int(unique_values.size)

        if array.dtype.kind == "b" or unique_count == 2:
            inferred[column_name] = "boolean"
            continue

        unique_ratio = unique_count / float(array.size)
        if array.dtype.kind in {"i", "u", "f"} and unique_ratio > unique_ratio_threshold:
            inferred[column_name] = "numerical"
        else:
            inferred[column_name] = "categorical"

    return inferred