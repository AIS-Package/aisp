"""ns: Negative Selection

The functions perform detector checks and utilize Numba decorators for Just-In-Time compilation
"""

import numpy.typing as npt
from numba import njit, types

from ..utils.distance import compute_metric_distance, hamming


@njit(
    [(
        types.boolean[:, :],
        types.boolean[:],
        types.float64
    )],
    cache=True
)
def check_detector_bnsa_validity(
    x_class: npt.NDArray,
    vector_x: npt.NDArray,
    aff_thresh: float
) -> bool:
    """
    Checks the validity of a candidate detector (vector_x) against samples from a class (x_class)
    using the Hamming distance. A detector is considered INVALID if its distance to any sample
    in ``x_class`` is less than or equal to ``aff_thresh``.

    Parameters
    ----------
    * x_class (``npt.NDArray``): Array containing the class samples. Expected shape: 
        (n_samples, n_features).
    * vector_x (``npt.NDArray``): Array representing the detector. Expected shape: (n_features,).
    * aff_thresh (``float``): Affinity threshold.

    Returns
    ----------
    * True if the detector is valid, False otherwise.
    """
    n = x_class.shape[1]
    if n != vector_x.shape[0]:
        return False

    for i in range(x_class.shape[0]):
        # Calculate the normalized Hamming Distance
        if hamming(x_class[i], vector_x) <= aff_thresh:
            return False
    return True


@njit(
    [(
        types.boolean[:],
        types.boolean[:, :, :],
        types.float64
    )],
    cache=True
)
def bnsa_class_prediction(
    features: npt.NDArray,
    class_detectors: npt.NDArray,
    aff_thresh: float
) -> int:
    """
    Defines the class of a sample from the non-self detectors.

    Parameters
    ----------
    * features (``npt.NDArray``): binary sample to be classified (shape: [n_features]).
    * class_detectors (``npt.NDArray``): Array containing the detectors of all classes
    (shape: [n_classes, n_detectors, n_features]).
    * aff_thresh (``float``): Affinity threshold that determines whether a detector recognizes the
    sample as non-self.

    Returns
    ----------
    * int: Index of the predicted class. Returns -1 if it is non-self for all classes.
    """
    n_classes, n_detectors, _ = class_detectors.shape
    best_class_idx = -1
    best_avg_distance = 0

    for class_index in range(n_classes):
        total_distance = 0.0
        class_found = True

        # Calculates the Hamming distance between the row and all detectors.
        for detector_index in range(n_detectors):
            # Calculates the normalized Hamming distance between the sample and the detector
            distance = hamming(features, class_detectors[class_index][detector_index])

            # If the distance is less than or equal to the threshold, the detector recognizes
            # the sample as non-self.
            if distance <= aff_thresh:
                class_found = False
                break
            total_distance += distance

        # if the sample is self for the class
        if class_found:
            avg_distance = total_distance / n_detectors
            # Choose the class with the largest average distance.
            if avg_distance > best_avg_distance:
                best_avg_distance = avg_distance
                best_class_idx = class_index

    return best_class_idx


@njit(
    [(
        types.float64[:, :], types.float64[:],
        types.float64, types.int32, types.float64
    )],
    cache=True
)
def check_detector_rnsa_validity(
    x_class: npt.NDArray,
    vector_x: npt.NDArray,
    threshold: float,
    metric: int,
    p: float
) -> bool:
    """
    Checks the validity of a candidate detector (vector_x) against samples from a class (x_class)
    using the Hamming distance. A detector is considered INVALID if its distance to any sample
    in ``x_class`` is less than or equal to ``aff_thresh``.

    Parameters
    ----------
    * x_class (``npt.NDArray``): Array containing the class samples. Expected shape: 
        (n_samples, n_features).
    * vector_x (``npt.NDArray``): Array representing the detector. Expected shape: (n_features,).
    * threshold (``float``): threshold.
    * metric (``str``): Distance metric to be used. Available options: 
    ["hamming", "cityblock", "minkowski", "euclidean"]
    * p (``float``): Parameter for the Minkowski distance (used only if `metric` 
    is "minkowski").

    Returns
    ----------
    * True if the detector is valid, False otherwise.
    """
    n = x_class.shape[1]
    if n != vector_x.shape[0]:
        return False

    for i in range(x_class.shape[0]):
        distance = compute_metric_distance(vector_x, x_class[i], metric, p)
        if distance <= threshold:
            return False
    return True
