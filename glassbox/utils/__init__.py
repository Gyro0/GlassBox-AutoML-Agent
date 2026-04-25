"""Utility functions used across the GlassBox package."""

from glassbox.utils.matrix import column_mean, column_std, column_variance, dot, transpose
from glassbox.utils.validation import check_array, check_is_fitted, check_consistent_length

__all__ = [
    "dot",
    "transpose",
    "column_mean",
    "column_variance",
    "column_std",
    "check_array",
    "check_is_fitted",
    "check_consistent_length",
]
