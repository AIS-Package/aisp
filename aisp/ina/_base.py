"""Base Class for Network Theory Algorithms."""

from abc import ABC
from typing import Optional

import numpy as np
from numpy import typing as npt

from ..base import BaseClusterer
from ..exceptions import FeatureDimensionMismatch
from ..utils.types import FeatureType


class BaseAiNet(BaseClusterer, ABC):
    """Abstract base class for AINet-based clustering algorithms."""

    @staticmethod
    def _check_and_raise_exceptions_fit(
        X: npt.NDArray
    ):
        """
        Verify the fit parameters and throw exceptions if the verification is not successful.

        Parameters
        ----------
        X : npt.NDArray
            Training array, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].

        Raises
        ------
        TypeError
            If X is not an ndarray or list.
        """
        if not isinstance(X, np.ndarray) and not isinstance(X, list):
            raise TypeError("X is not an ndarray or list.")

    @staticmethod
    def _check_and_raise_exceptions_predict(
        X: npt.NDArray,
        expected: int = 0,
        feature_type: FeatureType = "continuous-features"
    ) -> None:
        """
        Verify the predict parameters and throw exceptions if the verification is not successful.

        Parameters
        ----------
        X : npt.NDArray
            Input array for prediction, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].
        expected : int, default=0
            Expected number of features per sample (columns in X).
        feature_type : FeatureType, default="continuous-features"
            Specifies the type of features: "continuous-features", "binary-features",
            or "ranged-features".


        Raises
        ------
        TypeError
            If X is not a ndarray or list.
        FeatureDimensionMismatch
            If the number of features in X does not match the expected number.
        ValueError
            If feature_type is "binary-features" and X contains values other than 0 and 1.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError("X is not an ndarray or list")
        if expected != len(X[0]):
            raise FeatureDimensionMismatch(
                expected,
                len(X[0]),
                "X"
            )

        if feature_type != "binary-features":
            return

        # Checks if matrix X contains only binary samples. Otherwise, raises an exception.
        if not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )

    @staticmethod
    def _generate_random_antibodies(
        n_samples: int,
        n_features: int,
        feature_type: FeatureType = "continuous-features",
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
            or "ranged-features".
        bounds : np.ndarray
            Array (n_features, 2) with min and max per dimension.

        Returns
        -------
        npt.NDArray
            Array of shape (n_samples, n_features) containing the generated antibodies.
            Data type depends on the feature_type type (float for continuous/ranged, bool for
            binary).
        """
        if n_features <= 0:
            raise ValueError("Number of features must be greater than zero.")

        if feature_type == "binary-features":
            return np.random.randint(0, 2, size=(n_samples, n_features)).astype(np.bool_)
        if feature_type == "ranged-features" and bounds is not None:
            return np.random.uniform(low=bounds[0], high=bounds[1], size=(n_samples, n_features))

        return np.random.random_sample(size=(n_samples, n_features))
