"""Tests Utility functions to sanitizers variables"""

import numpy as np
import pytest

from aisp.utils.sanitizers import sanitize_bounds, sanitize_choice, sanitize_param, sanitize_seed


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

@pytest.mark.parametrize(
    "bounds, problem_size, expected_result, expected_exception",
    [
        ({'low': [0.0, 1.0], 'high': [10.0, 11.0]}, 2, {'low': [0., 1.], 'high': [10., 11.]}, None),
        ({'low': -5, 'high': 5}, 3, {'low': [-5., -5., -5.], 'high': [5., 5., 5.]}, None),
        ({'low': [0, 1], 'high': [10, 11]}, 2, {'low': [0., 1.], 'high': [10., 11.]}, None),
        ("not_dict", 2, None, ValueError),
        ({'low': [0, 1]}, 2, None, ValueError),
        ({'low': [0, 1, 2], 'high': [10, 11, 12]}, 2, None, ValueError),
        ({'low': [0, 1], 'high': [10, 'err']}, 2, None, TypeError),
        (None, 2, None, ValueError)
    ]
)
def test_sanitize_bounds(bounds, problem_size, expected_result, expected_exception):
    """
    Tests whether the sanitize_bounds function correctly returns.
    """
    if expected_exception:
        with pytest.raises(expected_exception):
            sanitize_bounds(bounds, problem_size)
    else:
        result = sanitize_bounds(bounds, problem_size)
        assert 'low' in result and 'high' in result
        assert np.array_equal(result['low'], expected_result['low'])
        assert np.array_equal(result['high'], expected_result['high'])