"""Utility functions for measuring accuracy and performance."""

from typing import Union

import numpy as np
import numpy.typing as npt


def accuracy_score(
    y_true: Union[npt.NDArray, list],
    y_pred: Union[npt.NDArray, list]
) -> float:
    """Calculate the accuracy score based on true and predicted labels.

    Parameters
    ----------
    y_true : Union[npt.NDArray, list]
        Ground truth (correct) labels. Expected to be of the same length as `y_pred`.
    y_pred : Union[npt.NDArray, list]
        Predicted labels. Expected to be of the same length as `y_true`.

    Returns
    -------
    accuracy : float
        The ratio of correct predictions to the total number of predictions.

    Raises
    ------
    ValueError
        If `y_true` or `y_pred` are empty or if they do not have the same length.

    Examples
    --------
    >>> import numpy as np
    >>> from aisp.utils.metrics import accuracy_score
    >>> y_true = [1, 1, 1, 1, 1]
    >>> y_pred = [1, 1, 1, 0, 0]
    >>> print(accuracy_score(y_true, y_pred))
    0.6
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("Division by zero: y_true cannot be an empty list or array.")
    if n != len(y_pred):
        raise ValueError(
            f"Error: The arrays must have the same size. Size of y_true: "
            f"{len(y_true)}, Size of y_pred: {len(y_pred)}"
        )
    return np.sum(np.array(y_true) == np.array(y_pred)) / n
