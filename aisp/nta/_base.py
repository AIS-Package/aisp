"""Base Class for Network Theory Algorithms."""

from abc import ABC
from typing import Literal

from numpy import typing as npt

import numpy as np

from ..base import BaseClassifier
from ..exceptions import FeatureDimensionMismatch


class BaseAiNet(BaseClassifier, ABC):
    """
    Base class for algorithm AiNet.
    """

    @staticmethod
    def _check_and_raise_exceptions_fit(
        X: npt.NDArray = None,
        y: npt.NDArray = None,
        algorithm: Literal[
            "continuous-features", "binary-features"
        ] = "continuous-features"
    ):
        """
        Verify the fit parameters and throw exceptions if the verification is not successful.

        Parameters
        ----------
        X : npt.NDArray
            Training array, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].
        y : npt.NDArray
            Array of target classes of ``X`` with [``N samples`` (lines)].
        algorithm : Literal["continuous-features", "binary-features"], default="continuous-features"
            Specifies the type of algorithm to use, depending on whether the input data has
            continuous or binary features.

        Raises
        ------
        TypeError:
            If X or y are not ndarrays or have incompatible shapes.
        ValueError
            If algorithm is binary-features and X contains values that are not composed only
            of 0 and 1.
        """
        if not isinstance(X, np.ndarray):
            if isinstance(X, list):
                X = np.array(X)
            else:
                raise TypeError("X is not an ndarray or list.")
        elif not isinstance(y, np.ndarray):
            if isinstance(y, list):
                y = np.array(y)
            else:
                raise TypeError("y is not an ndarray or list.")
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "X does not have the same amount of sample for the output classes in y."
            )

        if algorithm == "binary-features" and not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )

    @staticmethod
    def _check_and_raise_exceptions_predict(
        X: npt.NDArray = None,
        expected: int = 0,
        algorithm: Literal[
            "continuous-features", "binary-features"
        ] = "continuous-features"
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
        algorithm : Literal["continuous-features", "binary-features"], default="continuous-features"
            Specifies the type of algorithm to use, depending on whether the input data has
            continuous or binary features.

        Raises
        ------
        TypeError
            If X is not a ndarray or list.
        FeatureDimensionMismatch
            If the number of features in X does not match the expected number.
        ValueError
            If algorithm is binary-features and X contains values that are not composed only
            of 0 and 1.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError("X is not an ndarray or list")
        if expected != len(X[0]):
            raise FeatureDimensionMismatch(
                expected,
                len(X[0]),
                "X"
            )

        if algorithm != "binary-features":
            return

        # Checks if matrix X contains only binary samples. Otherwise, raises an exception.
        if not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )
