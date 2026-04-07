"""Exploratory data analysis utilities."""

from glassbox.eda.auto_typer import infer_column_types
from glassbox.eda.correlation import build_pearson_correlation_matrix
from glassbox.eda.outlier import iqr_outlier_handler
from glassbox.eda.profiler import profile_numeric_columns

__all__ = [
	"infer_column_types",
	"build_pearson_correlation_matrix",
	"iqr_outlier_handler",
	"profile_numeric_columns",
]
