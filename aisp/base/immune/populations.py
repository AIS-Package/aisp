"""Provide utility functions for generating antibody populations in immunological algorithms."""

from typing import Optional

import numpy as np
import numpy.typing as npt

from ...utils.types import FeatureTypeAll


def generate_random_antibodies(
    n_samples: int,
    n_features: int,
    feature_type: FeatureTypeAll = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray:
    """
    Generate a random antibody population.

    Parameters
    ----------
    n_samples : int
        Number of antibodies (samples) to generate.
    n_features : int
        Number of features (dimensions) for each antibody.
    feature_type : FeatureType, default="continuous-features"
        Specifies the type of features: "continuous-features", "binary-features",
        "ranged-features", or "permutation-features".
    bounds : np.ndarray
        Array (n_features, 2) with min and max per dimension.

    Returns
    -------
    npt.NDArray
        Array of shape (n_samples, n_features) containing the generated antibodies.
    """
    if n_features <= 0:
        raise ValueError("Number of features must be greater than zero.")

    if feature_type == "binary-features":
        return np.random.randint(0, 2, size=(n_samples, n_features)).astype(np.bool_)
    if feature_type == "ranged-features" and bounds is not None:
        return np.random.uniform(low=bounds[0], high=bounds[1], size=(n_samples, n_features))
    if feature_type == "permutation-features":
        return np.array(
            [np.random.permutation(n_features) for _ in range(n_samples)]
        ).astype(dtype=np.int64)

    return np.random.random_sample(size=(n_samples, n_features))
