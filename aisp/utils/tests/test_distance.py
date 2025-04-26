"""Test utility functions to measure normalized distance between vectors."""

import pytest
import numpy as np
from aisp.utils.distance import euclidean, hamming, cityblock, minkowski


@pytest.mark.parametrize(
    "u, v, expected_output",
    [
        (np.array([0.0, 0.0]), np.array([3.0, 4.0]), 5.0),
        (np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), 5.196),
    ],
    ids=[
        "Euclidean Distance - 2 dimensions",
        "Euclidean Distance - 3 dimensions",
    ]
)
def test_euclidean(u, v, expected_output):
    """Test for Euclidean distance"""
    result = round(euclidean(u, v), 3)
    assert result == expected_output


@pytest.mark.parametrize(
    "u, v, expected_output",
    [
        (np.array([1, 0, 1, 1]), np.array([1, 1, 0, 1]), 0.5),
        (np.array([True, False, True, False]), np.array([True, True, True, False]), 0.25),
    ],
    ids=[
        "Hamming distance between different binary integer vectors",
        "Hamming distance between different binary Boolean vectors",
    ]
)
def test_hamming(u, v, expected_output):
    """Hamming distance test"""
    result = hamming(u, v)
    assert result == expected_output


@pytest.mark.parametrize(
    "u, v, expected_output",
    [
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 3),
        (np.array([0, 0]), np.array([3, 4]), 3.5),
    ],
    ids=[
        "Manhattan Distance - 3 dimensions",
        "Manhattan Distance - 2 dimensions",
    ]
)
def test_cityblock(u, v, expected_output):
    """Manhattan (Cityblock) Distance Test"""
    result = cityblock(u, v)
    assert result == expected_output


@pytest.mark.parametrize(
    "u, v, p, expected_output",
    [
        (np.array([0, 0]), np.array([3, 4]), 1, 3.5),
        (np.array([0, 0]), np.array([3, 4]), 2, 2.5),
        (np.array([1, 2, 3]), np.array([4, 5, 6]), 3, 1.442),
    ],
    ids=[
        "Minkowski p=1 - Manhattan",
        "Minkowski p=2 - Euclidean",
        "Minkowski p=3 - Minkowski",
    ]
)
def test_minkowski(u, v, p, expected_output):
    """Minkowski distance test"""
    result = round(minkowski(u, v, p), 3)
    assert result == expected_output
