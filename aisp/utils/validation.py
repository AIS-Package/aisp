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
    - "binary": binary data (boolean True/False or integer 0/1)
    - "continuous": continuous data between 0.0 and 1.0 (float)
    - "ranged": numerical data with values outside the normalized range (float)

    Parameters
    ----------
    vector: npt.NDArray
        An array containing the data to be classified.

    Returns
    -------
    Literal["binary-features", "continuous-features", "ranged-features"]
        The classified data type of the vector.

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
