"""Contains functions responsible for validating data types."""

import numpy as np
import numpy.typing as npt

from .types import FeatureType
from ..exceptions import UnsupportedTypeError, FeatureDimensionMismatch


def detect_vector_data_type(vector: npt.NDArray) -> FeatureType:
    """
    Detect the type of data in a vector.

    The function detects if the vector contains data of type:
    - Binary features: boolean values or integers restricted to 0 and 1.
    - Continuous features: floating-point values in the normalized range [0.0, 1.0].
    - Ranged features: floating-point values outside the normalized range.

    Parameters
    ----------
    vector: npt.NDArray
        An array containing the data to be classified.

    Returns
    -------
    FeatureType
        The data type of the vector: "binary-features", "continuous-features", or "ranged-features".

    Raises
    ------
    UnsupportedTypeError
        If the data type of the vector is not supported by the function.
    """
    if vector.dtype == np.bool_:
        return "binary-features"

    if np.issubdtype(vector.dtype, np.integer) and np.isin(vector, [0, 1]).all():
        return "binary-features"

    if np.issubdtype(vector.dtype, np.floating):
        if np.all(vector >= 0.0) and np.all(vector <= 1.0):
            return "continuous-features"
        return "ranged-features"

    raise UnsupportedTypeError()


def check_array_type(x, name: str = "X") -> npt.NDArray:
    """Ensure X is a numpy array. Convert from list if needed.

    Parameters
    ----------
    x : Any
        Array, containing the samples and their characteristics,
        Shape: (n_samples, n_features).
    name : str, default='X'
        Variable name used in error messages.

    Returns
    -------
    npt.NDArray
        The converted or validated array.

    Raises
    ------
    TypeError:
        If X or y are not ndarrays or a list.
    """
    if isinstance(x, list):
        x = np.array(x)
    if not isinstance(x, np.ndarray):
        raise TypeError(f"{name} is not an ndarray or list.")
    return x


def check_shape_match(x: npt.NDArray, y: npt.NDArray):
    """Ensure X and y have compatible first dimensions.

    Parameters
    ----------
    x : npt.NDArray
        Array, containing the samples and their characteristics,
        Shape: (n_samples, n_features).
    y : npt.NDArray
        Array of target classes of ``x`` with (``n_samples``).

    Raises
    ------
    TypeError:
        If x or y have incompatible shapes.
    """
    if x.shape[0] != y.shape[0]:
        raise TypeError("X does not have the same number of samples as y.")


def check_feature_dimension(x: npt.NDArray, expected: int):
    """Ensure X has the expected number of features.

    Parameters
    ----------
    x : npt.NDArray
        Input array for prediction, containing the samples and their characteristics,
        Shape: (n_samples, n_features).
    expected : int, default=0
        Expected number of features per sample (columns in X).

    Raises
    ------
    FeatureDimensionMismatch
        If the number of features in X does not match the expected number.
    """
    if expected <= 0 or expected != len(x[0]):
        raise FeatureDimensionMismatch(expected, len(x[0]), "X")


def check_binary_array(x: npt.NDArray):
    """Ensure X contains only 0 and 1.

    Raises
    ------
    ValueError
        If the array contains values other than 0 and 1.
    """
    if not np.isin(x, [0, 1]).all():
        raise ValueError(
            "The array x contains values that are not composed only of 0 and 1."
        )
