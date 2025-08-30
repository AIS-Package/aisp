"""Utility functions for handling classes with multiple categories."""

from typing import Union

import numpy as np
import numpy.typing as npt


def slice_index_list_by_class(classes: Union[npt.NDArray, list], y: npt.NDArray) -> dict:
    """Separate indices of samples by class for targeted iteration.

    Parameters
    ----------
    classes: list or npt.NDArray
        list with unique classes.
    y : npt.NDArray
        Receives a ``y``[``N sample``] array with the output classes of the ``X`` sample array.

    Returns
    -------
    position_samples : dict
        A dictionary with the list of array positions(``y``), with the classes as key.

    Examples
    --------
    >>> import numpy as np
    >>> labels = ['a', 'b', 'c']
    >>> y = np.array(['a', 'c', 'b', 'a', 'c', 'b'])
    >>> slice_index_list_by_class(labels, y)
    {'a': [0, 3], 1: [2, 5], 2: [1, 4]}
    """
    position_samples = {}
    for _class_ in classes:
        # Gets the sample positions by class from y.
        position_samples[_class_] = np.flatnonzero(y == _class_).tolist()
    return position_samples
