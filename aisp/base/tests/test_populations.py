"""Utility function tests for generating antibody populations."""

import numpy as np
import pytest

from aisp.base.populations import generate_random_antibodies
from aisp.utils.types import FeatureTypeAll


@pytest.mark.parametrize(
    "n_samples, n_features, feature_type, bounds, expected_shape, expected_dtype",
    [
        (5, 5, "continuous-features", None, (5, 5), np.float64),
        (5, 5, "binary-features", None, (5, 5), np.bool_),
        (
            5, 5,
            "ranged-features",
            np.array([[1, 1, 1, 1, 1],
            [10, 10, 10, 10, 10]]),
            (5, 5),
            float
        ),
        (5, 5, "permutation-features", None, (5, 5), np.int64)
    ],
    ids=[
        "Generates a population with shape (5, 5) and continuous-features.",
        "Generates a population with shape (5, 5) and binary-features.",
        "Generates a population with shape (5, 5) and ranged-features.",
        "Generates a population with shape (5, 5) and permutation-features."
    ]
)
def test_generate_random_antibodies_valid_parametrized(
    n_samples,
    n_features,
    feature_type: FeatureTypeAll,
    bounds,
    expected_shape,
    expected_dtype
):
    antibodies = generate_random_antibodies(
        n_samples, n_features, feature_type, bounds
    )
    assert antibodies.shape == expected_shape
    assert np.issubdtype(antibodies.dtype, expected_dtype)
