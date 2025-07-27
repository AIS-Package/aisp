"""Tests Utility functions to validation"""

import numpy as np
import pytest
from aisp.exceptions import UnsupportedTypeError
from aisp.utils.validation import detect_vector_data_type


@pytest.mark.parametrize(
    "vector, expected",
    [
        (np.array([True, False, True], dtype=bool), "binary-features"),
        (np.array([0, 1, 0, 1], dtype=int), "binary-features"),
        (np.array([0.0, 0.5, 1.0, 0.9], dtype=float), "continuous-features"),
        (np.array([-1.0, 0.0, 1.5, 2.0], dtype=float), "ranged-features"),
    ],
    ids=[
        "binary-boolean",
        "binary-integers",
        "continuous-floats",
        "ranged-floats",
    ]
)
def test_detect_vector_data_type(vector, expected):
    """Tests different vector types and verifies the detected type."""
    assert detect_vector_data_type(vector) == expected


def test_unsupported_type():
    """Tests array with unsupported type that should raise exception."""
    vec = np.array(["a", "b", "c"])
    with pytest.raises(UnsupportedTypeError):
        detect_vector_data_type(vec)
