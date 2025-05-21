"""Utility functions for normalized distance between arrays with numba decorators."""

import numpy as np
import numpy.typing as npt
from numba import njit, types

EUCLIDEAN = 0
MANHATTAN = 1
MINKOWSKI = 2
HAMMING = 3


@njit([(types.boolean[:], types.boolean[:])], cache=True)
def hamming(u: npt.NDArray[np.bool_], v: npt.NDArray[np.bool_]) -> np.float64:
    """Calculate the normalized Hamming distance between two points.
    
    ((x₁ ≠ x₂) + (y₁ ≠ y₂) + ... + (yn ≠ yn)) / n

    Parameters
    ----------
    u : npt.NDArray[np.bool_]
        Coordinates of the first point.
    v : npt.NDArray[np.bool_]
        Coordinates of the second point.

    Returns
    -------
    Distance : np.float64
        Distance : float``) between the two points.
    """
    n = len(u)
    if n == 0:
        return 0.0

    return np.sum(u != v) / n


@njit()
def euclidean(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> np.float64:
    """Calculate the normalized Euclidean distance between two points.
    
    √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²)

    Parameters
    ----------
    u : npt.NDArray[np.float64]
        Coordinates of the first point.
    v : npt.NDArray[np.float64]
        Coordinates of the second point.

    Returns
    -------
    distance : np.float64
        Distance : float``) between the two points.
    """
    return np.linalg.norm(u - v)


@njit()
def cityblock(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> np.float64:
    """Calculate the normalized Manhattan distance between two points.
    
    (|x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) / n

    Parameters
    ----------
    u : npt.NDArray[np.float64]
        Coordinates of the first point.
    v : npt.NDArray[np.float64]
        Coordinates of the second point.

    Returns
    -------
    distance : np.float64
        Distance (``float``) between the two points.
    """
    n = len(u)
    if n == 0:
        return -1.0

    return np.sum(np.abs(u - v)) / n


@njit()
def minkowski(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64], p: float = 2.0) -> np.float64:
    """Calculate the normalized Minkowski distance between two points.
    
    (( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.) / n

    Parameters
    ----------
    u : npt.NDArray[np.float64]
        Coordinates of the first point.
    v : npt.NDArray[np.float64]
        Coordinates of the second point.
    p : float
        The p parameter defines the type of distance to be calculated:

        - p = 1: **Manhattan** distance — sum of absolute differences.
        - p = 2: **Euclidean** distance — sum of squared differences (square root).
        - p > 2: **Minkowski** distance with an increasing penalty as p increases.

    Returns
    -------
    np.float64
        Distance : float``) between the two points.
    """
    n = len(u)
    if n == 0:
        return -1.0

    return (np.sum(np.abs(u - v) ** p) ** (1 / p)) / n


@njit(
    [(
        types.float64[:], types.float64[:],
        types.int32, types.float64
    )],
    cache=True
)
def compute_metric_distance(
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    metric: int,
    p: np.float64 = 2.0
) -> np.float64:
    """Calculate the distance between two points by the chosen metric.

    Parameters
    ----------
    u : npt.NDArray[np.float64]
        Coordinates of the first point.
    v : npt.NDArray[np.float64]
        Coordinates of the second point.
    metric : int
        Distance metric to be used. Available options:  [0 (Euclidean), 1 (Manhattan), 
        2 (Minkowski)]
    p : float, default=2.0
        Parameter for the Minkowski distance (used only if `metric` is "minkowski").

    Returns
    -------
    np.float64
        Distance (``float``) between the two points with the selected metric.
    """
    if metric == MANHATTAN:
        return cityblock(u, v)
    if metric == MINKOWSKI:
        return minkowski(u, v, p)

    return euclidean(u, v)


@njit(
    [(
        types.float64[:, :], types.float64[:],
        types.int32, types.float64
    )],
    cache=True
)
def min_distance_to_class_vectors(
    x_class: npt.NDArray[np.float64],
    vector_x: npt.NDArray[np.float64],
    metric: int,
    p: float = 2.0
) -> float:
    """Calculate the minimum distance between an input vector and the vectors of a class.

    Parameters
    ----------
    x_class : npt.NDArray[np.float64]
        Array containing the class vectors to be compared with the input vector. Expected shape:
        (n_samples, n_features).
    vector_x : npt.NDArray[np.float64]
        Vector to be compared with the class vectors. Expected shape: (n_features,).
    metric : int
        Distance metric to be used. Available options: ["hamming", "cityblock", "minkowski",
        "euclidean"]
    p : float, default=2.0
        Parameter for the Minkowski distance (used only if `metric` is "minkowski").

    Returns
    -------
    min_distance : float:
        The minimum distance calculated between the input vector and the class vectors. 
        Returns -1.0 if the input dimensions are incompatible.
    """
    n = x_class.shape[1]
    if n != vector_x.shape[0]:
        return -1.0

    min_distance = np.inf
    for i in range(x_class.shape[0]):
        distance = compute_metric_distance(vector_x, x_class[i], metric, p)
        min_distance = min(min_distance, distance)

    return min_distance


def get_metric_code(metric: str) -> int:
    """Get the numeric code associated with a distance metric.

    Parameters
    ----------
    metric : str
        Name of the metric. Can be "euclidean", "manhattan", "minkowski" or "hamming".

    Raises
    ------
    ValueError
        If the metric provided is not supported.

    Returns
    -------
    int
        Numeric code corresponding to the metric.
    """
    metric_map = {
        "euclidean": EUCLIDEAN,
        "manhattan": MANHATTAN,
        "minkowski": MINKOWSKI,
        "hamming": HAMMING
    }

    normalized_metric = metric.strip().lower()

    if normalized_metric not in metric_map:
        supported = "', '".join(metric_map.keys())
        raise ValueError(f"Unknown metric: '{metric}'. Supported: {supported}")

    return metric_map[normalized_metric]
