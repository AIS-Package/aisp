"""Base Class for Clonal Selection Algorithm."""

from abc import ABC

import numpy as np
import numpy.typing as npt

from ..exceptions import FeatureDimensionMismatch
from ..utils.types import FeatureType
from ..base import BaseClassifier


class BaseAIRS(BaseClassifier, ABC):
    """
    Base class for algorithm AIRS.

    The base class contains functions that are used by more than one class in the package, and
    therefore are considered essential for the overall functioning of the system.
    """

    @staticmethod
    def _check_and_raise_exceptions_fit(
        X: npt.NDArray,
        y: npt.NDArray
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

        Raises
        ------
        TypeError:
            If X or y are not ndarrays or have incompatible shapes.
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
            Specifies the type of feature_type to use, depending on whether the input data has
            continuous or binary features.

        Raises
        ------
        TypeError
            If X is not a ndarray or list.
        FeatureDimensionMismatch
            If the number of features in X does not match the expected number.
        ValueError
            If feature_type is binary-features and X contains values that are not composed only
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

        if feature_type != "binary-features":
            return

        # Checks if matrix X contains only binary samples. Otherwise, raises an exception.
        if not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )
