"""Utility functions for handling classes with multiple categories."""

from collections import Counter
from heapq import nlargest
from typing import Union, Callable, List, Tuple

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


def predict_knn_affinity(
    X: npt.NDArray,
    k: int,
    all_cell_vectors: List[Tuple[Union[str, int], npt.NDArray]],
    affinity_func: Callable[[npt.NDArray, npt.NDArray], float]
) -> npt.NDArray:
    """Predict classes using k-nearest neighbors and trained cells.

    Parameters
    ----------
    X: npt.NDArray
        Input data to be classified.
    k: int
        Number of nearest neighbors to consider for prediction.
    all_cell_vectors: List[Tuple[Union[str, int], npt.NDArray]]
        List of tuples (class_name, cell[np.ndarray]).
    affinity_func: Callable[[npt.NDArray, npt.NDArray], float]
        Function that takes two vectors and returns an affinity value.

    Returns
    -------
    npt.NDArray
        Array of predicted labels for each sample in X, based on the k nearest neighbors.
    """
    c: list = []

    for line in X:
        label_stim_list = [
            (class_name, affinity_func(cell, line))
            for class_name, cell in all_cell_vectors
        ]
        # Create the list with the k nearest neighbors and select the class with the most votes
        k_nearest = nlargest(k, label_stim_list, key=lambda x: x[1])
        votes = Counter(label for label, _ in k_nearest)
        c.append(votes.most_common(1)[0][0])

    return np.asarray(c)
