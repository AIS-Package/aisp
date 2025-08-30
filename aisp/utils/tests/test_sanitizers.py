"""Tests Utility functions to sanitizers variables"""

import pytest

from aisp.utils.sanitizers import sanitize_choice, sanitize_param, sanitize_seed


@pytest.mark.parametrize(
    "input_value, valid_choices, default, expected_output",
    [
        ("manhattan", {"manhattan", "euclidean"}, "default", "manhattan"),
        ("test", {"manhattan", "euclidean"}, "default", "default"),
        (None, {"manhattan", "euclidean"}, "default", "default")
    ],
    ids=[
        "When input is 'manhattan' and is in valid_choices",
        "When input is not in valid_choices: must use default",
        "When input is None: must use default"
    ]
)
def test_sanitize_choice(input_value, valid_choices, default, expected_output):
    """
    Tests whether the sanitize_choice function correctly returns the value or the default.
    """
    result = sanitize_choice(input_value, valid_choices, default)
    assert result == expected_output


@pytest.mark.parametrize(
    "input_value, default_value, condition, expected_output",
    [
        (10, 5, lambda x: x > 0, 10),
        (-1, 5, lambda x: x > 0, 5),
        (0, 5, lambda x: x > 0, 5),
        (None, 5, lambda x: x is not None, 5)
    ],
    ids=[
        "Positive value meets the condition",
        "Negative value does not meet the condition, returns default",
        "Zero does not meet the condition, returns default",
        "None does not meet the condition, returns default"
    ]
)
def test_sanitize_param(input_value, default_value, condition, expected_output):
    """
    Tests whether the sanitize_param function correctly returns.
    """
    result = sanitize_param(input_value, default_value, condition)
    assert result == expected_output


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (42, 42),
        (-1, None),
        ("random", None),
        (None, None),
    ],
    ids=[
        "Valid seed: positive number",
        "Negative seed: should return None",
        "Invalid seed: string, should return None",
        "Null seed: should return None"
    ]
)
def test_sanitize_seed(input_value, expected_output):
    """
    Tests whether the sanitize_seed function correctly returns the seed or None.
    """
    result = sanitize_seed(input_value)
    assert result == expected_output
