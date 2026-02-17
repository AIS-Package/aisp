"""nsa - Negative Selection.

The functions perform detector checks and utilize Numba decorators for Just-In-Time compilation
"""

import numpy as np
import numpy.typing as npt
from numba import njit, types

from ..utils.distance import compute_metric_distance, hamming


@njit([(types.boolean[:, :], types.boolean[:], types.float64)], cache=True)
def check_detector_bnsa_validity(
    x_class: npt.NDArray[np.bool_],
    vector_x: npt.NDArray[np.bool_],
    aff_thresh: float
) -> bool:
    """
    Check the validity of a candidate detector using the Hamming distance.

    A detector is considered INVALID if its distance to any sample in ``x_class`` is less than or
    equal to ``aff_thresh``.

    Parameters
    ----------
    x_class : npt.NDArray[np.bool_]
        Array containing the class samples. Expected shape:  (n_samples, n_features).
    vector_x : npt.NDArray[np.bool_]
        Array representing the detector. Expected shape: (n_features,).
    aff_thresh : float
        Affinity threshold.

    Returns
    -------
    valid : bool
        True if the detector is valid, False otherwise.
    """
    n = x_class.shape[1]
    if n != vector_x.shape[0]:
        return False

    for i in range(x_class.shape[0]):
        if hamming(x_class[i], vector_x) <= aff_thresh:
            return False
    return True


@njit([(types.boolean[:], types.boolean[:, :, :], types.float64)], cache=True)
def bnsa_class_prediction(
    features: npt.NDArray[np.bool_],
    class_detectors: npt.NDArray[np.bool_],
    aff_thresh: float,
) -> int:
    """Define the class of a sample from the non-self detectors.

    Parameters
    ----------
    features : npt.NDArray[np.bool_]
        binary sample to be classified (shape: [n_features]).
    class_detectors : npt.NDArray[np.bool_]
        Array containing the detectors of all classes (shape: [n_classes, n_detectors, n_features]).
    aff_thresh : float
        Affinity threshold that determines whether a detector recognizes the sample as non-self.

    Returns
    -------
    best_class_index : int
        Index of the predicted class. Returns -1 if it is non-self for all classes.
    """
    n_classes, n_detectors, _ = class_detectors.shape
    best_class_idx = -1
    best_avg_distance = 0.0

    for class_index in range(n_classes):
        total_distance = 0.0
        class_found = True

        for detector_index in range(n_detectors):
            distance = hamming(features, class_detectors[class_index][detector_index])

            # If the distance is less than or equal to the threshold, the detector recognizes
            # the sample as non-self.
            if distance <= aff_thresh:
                class_found = False
                break
            total_distance += distance

        if class_found:
            avg_distance = total_distance / n_detectors
            # Choose the class with the largest average distance.
            if avg_distance > best_avg_distance:
                best_avg_distance = avg_distance
                best_class_idx = class_index

    return best_class_idx


@njit(
    [
        (
            types.float64[:, :],
            types.float64[:],
            types.float64,
            types.int32,
            types.float64,
        )
    ],
    cache=True,
)
def check_detector_rnsa_validity(
    x_class: npt.NDArray[np.float64],
    vector_x: npt.NDArray[np.float64],
    threshold: float,
    metric: int,
    p: float,
) -> bool:
    """Check the validity of a candidate detector using the Hamming distance.

    A detector is considered INVALID if its distance to any sample  in ``x_class`` is less than
    or equal to ``aff_thresh``.

    Parameters
    ----------
    x_class : npt.NDArray[np.float64]
        Array containing the class samples. Expected shape: (n_samples, n_features).
    vector_x : npt.NDArray[np.float64]
        Array representing the detector. Expected shape: (n_features,).
    threshold : float
        threshold.
    metric : int
        Distance metric to be used. Available options: [0 (Euclidean), 1 (Manhattan),
        2 (Minkowski)].
    p : float
        Parameter for the Minkowski distance (used only if `metric` is "minkowski").

    Returns
    -------
    valid : bool
        True if the detector is valid, False otherwise.
    """
    n = x_class.shape[1]
    if n != vector_x.shape[0]:
        return False

    for i in range(x_class.shape[0]):
        distance = compute_metric_distance(vector_x, x_class[i], metric, p)
        if distance <= threshold:
            return False
    return True
