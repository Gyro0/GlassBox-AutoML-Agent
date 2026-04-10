"""Automatic column type inference for EDA workflows."""

from __future__ import annotations

from typing import Any

import numpy as np


def _is_boolean_like(unique_values: np.ndarray) -> bool:
    """Return True when values are a canonical boolean representation."""
    if unique_values.size == 0:
        return False

    if unique_values.dtype.kind == "b":
        return True

    allowed_numeric = {0, 1}
    allowed_text = {"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"}

    normalized: set[str] = set()
    for value in unique_values.tolist():
        if isinstance(value, (bool, np.bool_)):
            normalized.add("1" if bool(value) else "0")
            continue
        if isinstance(value, (int, np.integer, float, np.floating)):
            if float(value).is_integer() and int(value) in allowed_numeric:
                normalized.add(str(int(value)))
                continue
            return False

        text = str(value).strip().lower()
        if text in allowed_text:
            normalized.add(text)
            continue
        return False

    return normalized.issubset(allowed_text) and len(normalized) <= 2


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

        if _is_boolean_like(unique_values):
            inferred[column_name] = "boolean"
            continue

        unique_ratio = unique_count / float(array.size)
        if array.dtype.kind in {"i", "u", "f"} and unique_ratio > unique_ratio_threshold:
            inferred[column_name] = "numerical"
        else:
            inferred[column_name] = "categorical"

    return inferred