"""Tests Utility functions for measuring accuracy and performance."""
import pytest

from aisp.utils.metrics import accuracy_score


def test_accuracy_score_empty_lists():
    """
    Tests whether the accuracy_score function raises an error when given empty lists.
    """
    with pytest.raises(
            ValueError,
            match="Division by zero: y_true cannot be an empty list or array."
    ):
        accuracy_score([], [])


def test_accuracy_score_perfect_predictions():
    """
    Tests whether the accuracy_score function returns 1.0 when all predictions are correct.
    """
    y_true = ['a', 'b', 'a', 'a']
    y_pred = ['a', 'b', 'a', 'a']
    result = accuracy_score(y_true, y_pred)
    assert result == 1


def test_accuracy_score_different_sizes():
    """
    Tests whether the accuracy_score function raises an error when lists have different sizes.
    """
    y_true = [1, 0, 1]
    y_pred = [1, 0]
    with pytest.raises(
            ValueError,
            match=f"Error: The arrays must have the same size. Size of y_true: "
                  f"{len(y_true)}, Size of y_pred: {len(y_pred)}"
    ):
        accuracy_score(y_true, y_pred)
