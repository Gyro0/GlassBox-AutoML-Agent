"""Classification metrics computed from a manual confusion matrix."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def build_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[Any] | None = None,
) -> tuple[np.ndarray, list[Any]]:
    """Build a K x K confusion matrix where rows are true and columns are predicted."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples.")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred])).tolist()

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    n_classes = len(labels)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for truth, prediction in zip(y_true, y_pred):
        matrix[label_to_index[truth], label_to_index[prediction]] += 1

    return matrix, labels


def _safe_divide(numerator: float, denominator: float) -> float:
    if np.isclose(denominator, 0.0):
        return 0.0
    return numerator / denominator


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: Literal["binary", "macro"] = "binary",
    positive_label: Any | None = None,
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """Return accuracy, precision, recall, and F1 from a manual confusion matrix."""
    confusion_matrix, class_labels = build_confusion_matrix(y_true, y_pred, labels=labels)

    total = float(np.sum(confusion_matrix))
    accuracy = _safe_divide(float(np.trace(confusion_matrix)), total)

    if average == "macro":
        precisions: list[float] = []
        recalls: list[float] = []
        f1_scores: list[float] = []

        for idx in range(len(class_labels)):
            tp = float(confusion_matrix[idx, idx])
            fp = float(np.sum(confusion_matrix[:, idx]) - tp)
            fn = float(np.sum(confusion_matrix[idx, :]) - tp)

            precision = _safe_divide(tp, tp + fp)
            recall = _safe_divide(tp, tp + fn)
            f1 = _safe_divide(2.0 * precision * recall, precision + recall)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        precision_value = float(np.mean(precisions))
        recall_value = float(np.mean(recalls))
        f1_value = float(np.mean(f1_scores))
    elif average == "binary":
        if len(class_labels) != 2:
            raise ValueError("Binary averaging requires exactly two classes.")

        pos_label = positive_label
        if pos_label is None:
            pos_label = 1 if 1 in class_labels else class_labels[-1]
        if pos_label not in class_labels:
            raise ValueError("positive_label must be present in class labels.")

        pos_idx = class_labels.index(pos_label)
        tp = float(confusion_matrix[pos_idx, pos_idx])
        fp = float(np.sum(confusion_matrix[:, pos_idx]) - tp)
        fn = float(np.sum(confusion_matrix[pos_idx, :]) - tp)

        precision_value = _safe_divide(tp, tp + fp)
        recall_value = _safe_divide(tp, tp + fn)
        f1_value = _safe_divide(2.0 * precision_value * recall_value, precision_value + recall_value)
    else:
        raise ValueError("average must be either 'binary' or 'macro'.")

    return {
        "accuracy": accuracy,
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "confusion_matrix": confusion_matrix,
        "labels": class_labels,
    }