"""cs: Clonal Selection

The functions perform detector checks and utilize Numba decorators for Just-In-Time compilation
"""
import numpy as np
import numpy.typing as npt
from numba import njit, types

from ..utils.distance import compute_metric_distance, hamming


@njit([(types.float64[:, :], types.int64, types.float64)], cache=True)
def continuous_affinity_threshold(
    x_class: npt.NDArray[np.float64],
    metric: int,
    p: float
) -> float:
    """
    Esta função calcula o limite de afinidade com base na afinidade média entre
    instâncias de treinamento, onde aᵢ e aⱼ são um par de antígenos, e a afinidade

    A afinidade entre dois vetores u e v é definida como:
        affinity = 1 - (distance / (1 + distance))

    Parameters
    ----------
    * x_class (np.ndarray): Vetores da mesma classe (shape: [n amostras, n dimensões]).
    * metric (int): Código da métrica a ser usada [0 (Euclidean), 1 (Manhattan), 2 (Minkowski)].
    * p (float): Parâmetro para distância de Minkowski (usado se metric==MINKOWSKI).

    Returns
    ----------
    * float: Threshold de afinidade para os vetores fornecidos.
    """
    n, _ = x_class.shape
    if n == 0:
        return -1.0

    sum_affinity = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            distance = compute_metric_distance(x_class[i], x_class[j], metric, p)
            sum_affinity += 1.0 - (distance / (1.0 + distance))

    return 1.0 - (sum_affinity / ((n * (n - 1)) / 2))


@njit([(types.float64(types.boolean[:, :]))], cache=True)
def binary_affinity_threshold(x_class: npt.NDArray[np.bool_]) -> float:
    """
    Esta função calcula o limite de afinidade com base na afinidade média entre
    instâncias de treinamento com dados binários, onde aᵢ e aⱼ são um par de antígenos,
    e a afinidade

    Usa distância de Hamming normalizada, convertida em afinidade:
        affinity = 1 - (distance / (1 + distance))


    Parameters
    ----------
    * x_class (np.ndarray): Vetores binários (shape: [n amostras, n bits]).

    Returns
    ----------
    * float: Threshold de afinidade para os vetores fornecidos.
    """
    n, _ = x_class.shape
    if n == 0:
        return -1.0

    sum_affinity = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            distance = hamming(x_class[i], x_class[j])
            sum_affinity += 1.0 - (distance / (1.0 + distance))

    return 1.0 - (sum_affinity / ((n * (n - 1)) / 2))
