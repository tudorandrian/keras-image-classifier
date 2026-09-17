"""Classification metrics in plain NumPy, small enough to check by hand."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

IntMatrix = NDArray[np.int64]


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> IntMatrix:
    """Rows are the true class, columns the predicted class."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return matrix


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def report(matrix: IntMatrix, classes: list[str]) -> dict[str, Any]:
    """Accuracy, macro F1, the majority-class baseline and per-class precision, recall, F1.

    A class that was never predicted gets precision 0 instead of an error: that is
    the number a reader needs to see.
    """
    total = int(matrix.sum())
    per_class: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(classes):
        hits = float(matrix[index, index])
        precision = _ratio(hits, float(matrix[:, index].sum()))
        recall = _ratio(hits, float(matrix[index, :].sum()))
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2 * precision * recall, precision + recall),
            "support": int(matrix[index, :].sum()),
        }
    return {
        "samples": total,
        "accuracy": _ratio(float(np.trace(matrix)), total),
        "macro_f1": float(np.mean([float(entry["f1"]) for entry in per_class.values()])),
        "majority_baseline": _ratio(float(matrix.sum(axis=1).max()), total),
        "per_class": per_class,
    }
