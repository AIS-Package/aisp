"""Tests Utility functions for handling classes with multiple categories."""

import numpy as np
import pytest

from aisp.utils.multiclass import slice_index_list_by_class, predict_knn_affinity


def test_slice_index_list_by_class():
    """
    Tests whether the slice_index_list_by_class function correctly returns indexes.
    """
    classes = ["a", "b"]
    y = np.array(["a", "b", "a", "a"])
    expected = {"a": [0, 2, 3], "b": [1]}
    result = slice_index_list_by_class(classes, y)
    assert result == expected


@pytest.mark.parametrize(
    "X, k, all_cells, affinity, expected",
    [
        (
            np.array([[1, 1], [4, 4]]),
            3,
            [
                ("A", np.array([1, 1])),
                ("A", np.array([2, 2])),
                ("B", np.array([3, 3])),
                ("B", np.array([4, 4])),
                ("C", np.array([5, 5])),
            ],
            lambda a, b: -np.linalg.norm(a - b),
            ["A", "B"],
        ),
        (
            np.array([[1, 1]]),
            3,
            [
                ("A", np.array([1, 1])),
                ("B", np.array([1, 1])),
            ],
            lambda a, b: -np.linalg.norm(a - b),
            ["A"],
        ),
        (
            np.array([[1, 1], [4, 4]]),
            3,
            [
                (1, np.array([1, 1])),
                (1, np.array([2, 2])),
                (2, np.array([3, 3])),
                (2, np.array([4, 4])),
                (3, np.array([5, 5])),
            ],
            lambda a, b: -np.linalg.norm(a - b),
            [1, 2],
        ),
    ],
    ids=["Multiple samples.", "Tie votes counter.", "Numeric labels."],
)
def test_predict_knn_affinity(X, k, all_cells, affinity, expected):
    """Test different KNN predictions."""
    pred = predict_knn_affinity(X, k, all_cells, affinity)
    assert pred.tolist() == expected
    assert len(X) == len(pred)
