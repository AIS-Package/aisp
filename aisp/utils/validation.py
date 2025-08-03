"""Contains functions responsible for validating data types."""

import numpy as np
import numpy.typing as npt

from .types import FeatureType
from ..exceptions import UnsupportedTypeError


def detect_vector_data_type(
    vector: npt.NDArray
) -> FeatureType:
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
    str
        The data type of the vector: "binary-features", "continuous-features", or "ranged-features".

    Raises
    ------
    UnsupportedDataTypeError
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
