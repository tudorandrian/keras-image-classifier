from __future__ import annotations

import numpy as np
import pytest

from keras_image_classifier.metrics import confusion_matrix, report

# Worked by hand: 10 samples, classes a, b, c.
Y_TRUE = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
Y_PRED = [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]


def test_confusion_matrix_has_true_rows_and_predicted_columns() -> None:
    matrix = confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert matrix.tolist() == [[3, 1, 0], [0, 2, 1], [1, 0, 2]]


def test_report_matches_the_hand_calculation() -> None:
    result = report(confusion_matrix(Y_TRUE, Y_PRED, 3), ["a", "b", "c"])
    assert result["samples"] == 10
    assert result["accuracy"] == pytest.approx(0.7)
    assert result["majority_baseline"] == pytest.approx(0.4)
    a = result["per_class"]["a"]
    assert a == pytest.approx({"precision": 0.75, "recall": 0.75, "f1": 0.75, "support": 4})
    b = result["per_class"]["b"]
    assert b == pytest.approx({"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3, "support": 3})
    assert result["macro_f1"] == pytest.approx((0.75 + 2 / 3 + 2 / 3) / 3)


def test_a_class_that_is_never_predicted_scores_zero_without_warnings() -> None:
    result = report(confusion_matrix([0, 0, 1, 1], [0, 0, 0, 0], 2), ["seen", "missed"])
    assert result["per_class"]["missed"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 2,
    }
    assert result["accuracy"] == 0.5
    assert result["majority_baseline"] == 0.5


def test_an_empty_matrix_gives_zeros() -> None:
    result = report(np.zeros((2, 2), dtype=np.int64), ["a", "b"])
    assert (result["samples"], result["accuracy"], result["macro_f1"]) == (0, 0.0, 0.0)
