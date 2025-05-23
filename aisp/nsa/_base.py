"""Base Class for Negative Selection Algorithm."""

from abc import ABC
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from ..base import BaseClassifier
from ..exceptions import FeatureDimensionMismatch


class BaseNSA(BaseClassifier, ABC):
    """
    Base class containing functions used by multiple classes in the package.

    These functions are essential for the overall functioning of the system.
    """

    @staticmethod
    def _check_and_raise_exceptions_fit(
        X: npt.NDArray = None,
        y: npt.NDArray = None,
        _class_: Literal["RNSA", "BNSA"] = "RNSA",
    ) -> None:
        """Verify fit function parameters.

        Throw exceptions if the verification fails.

        Parameters
        ----------
        * X : npt.NDArray
            Training array, containing the samples and their characteristics, [``N samples`` (
            rows)][``N features`` (columns)].
        * y : npt.NDArray
            Array of target classes of ``X`` with [``N samples`` (lines)].
        * _class_ : Literal[RNSA, BNSA], default="RNSA"
            Current class.

        Raises
        ------
        TypeError
            If X or y are not ndarrays or have incompatible shapes.
        ValueError
            If _class_ is BNSA and X contains values that are not composed only of 0 and 1.
        """
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)

        if not isinstance(X, np.ndarray):
            raise TypeError("X is not an ndarray or list.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y is not an ndarray or list.")

        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "X does not have the same amount of sample for the output classes in y."
            )

        if _class_ == "BNSA" and not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )

    @staticmethod
    def _check_and_raise_exceptions_predict(
        X: npt.NDArray = None,
        expected: int = 0,
        _class_: Literal["RNSA", "BNSA"] = "RNSA",
    ) -> None:
        """Verify predict function parameters.

        Throw exceptions if the verification fails.

        Parameters
        ----------
        X : npt.NDArray
            Input array for prediction, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].
        expected : int
            Expected number of features per sample (columns in X).
        _class_ : Literal[RNSA, BNSA], default="RNSA"
            Current class. Defaults to 'RNSA'.

        Raises
        ------
        TypeError
            If X is not an numpy.ndarray or list.
        FeatureDimensionMismatch
            If the number of features in X does not match the expected number.
        ValueError
            If _class_ is BNSA and X contains values that are not composed only of 0 and 1.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError("X is not an ndarray or list")
        if expected != len(X[0]):
            raise FeatureDimensionMismatch(expected, len(X[0]), "X")

        if _class_ != "BNSA":
            return

        # Checks if matrix X contains only binary samples. Otherwise, raises an exception.
        if not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )


@dataclass(slots=True)
class Detector:
    """
    Represents a non-self detector of the RNSA class.

    Attributes
    ----------
    position : npt.NDArray[np.float64]
        Detector feature vector.
    radius : float, optional
        Detector radius, used in the V-detector algorithm.
    """

    position: npt.NDArray[np.float64]
    radius: Optional[float] = None
