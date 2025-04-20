"""Utility functions for normalized distance between arrays with numba decorators."""

import numpy as np
import numpy.typing as npt
from numba import njit


@njit(cache=True)
def hamming(u: npt.NDArray, v: npt.NDArray):
    """
    Function to calculate the normalized Hamming distance between two points.
    
    ((x₁ ≠ x₂) + (y₁ ≠ y₂ + ... + (yn ≠ yn)) / n

    Parameters
    ----------
    * u (``npt.NDArray``): Coordinates of the first point.
    * v (``npt.NDArray``): Coordinates of the second point.

    returns
    ----------
    * Distance (``float``) between the two points.

    ----

    Função para calcular a distância de Hamming normalizada entre dois pontos.

    ((x₁ ≠ x₂) + (y₁ ≠ y₂ + ... + (yn ≠ yn)) / n

    Parameters
    ----------
    * u (``npt.NDArray``): Coordenadas do primeiro ponto.
    * v (``npt.NDArray``): Coordenadas do segundo ponto.

    returns
    ----------
    * Distância (``float``) entre os dois pontos.
    """
    return np.sum(u != v) / len(u)

@njit(cache=True)
def euclidean(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]):
    """
    Function to calculate the normalized Euclidean distance between two points.
    
    √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²)

    Parameters
    ----------
    * u (``npt.NDArray``): Coordinates of the first point.
    * v (``npt.NDArray``): Coordinates of the second point.

    returns
    ----------
    * Distance (``float``) between the two points.

    ----

    Função para calcular a distância euclidiana normalizada entre dois pontos.

    √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²)

    Parameters
    ----------
    * u (``npt.NDArray``): Coordenadas do primeiro ponto.
    * v (``npt.NDArray``): Coordenadas do segundo ponto.

    returns
    ----------
    * Distância (``float``) entre os dois pontos.
    """
    return np.linalg.norm(u - v)

@njit(cache=True)
def cityblock(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]):
    """
    Function to calculate the normalized Manhattan distance between two points.
    
    (|x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) / n

    Parameters
    ----------
    * u (``npt.NDArray``): Coordinates of the first point.
    * v (``npt.NDArray``): Coordinates of the second point.

    returns
    ----------
    * Distance (``float``) between the two points.

    -----

    Função para calcular a distância Manhattan normalizada entre dois pontos.

    (|x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) / n

    Parameters
    ----------
    * u (``npt.NDArray``): Coordenadas do primeiro ponto.
    * v (``npt.NDArray``): Coordenadas do segundo ponto.

    returns
    ----------
    * Distância (``float``) entre os dois pontos.
    """
    return np.sum(np.abs(u - v)) / len(u)

@njit(cache=True)
def minkowski(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64], p: float = 2):
    """
    Function to calculate the normalized Minkowski distance between two points.
    
    (( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.) / n

    Parameters
    ----------
    * u (``npt.NDArray``): Coordinates of the first point.
    * v (``npt.NDArray``): Coordinates of the second point.
    * p float: The p parameter defines the type of distance to be calculated:
        - p = 1: **Manhattan** distance — sum of absolute differences.
        - p = 2: **Euclidean** distance — sum of squared differences (square root).
        - p > 2: **Minkowski** distance with an increasing penalty as p increases.

    returns
    ----------
    * Distance (``float``) between the two points.

    -----

    Função para calcular a distância de Minkowski normalizada entre dois pontos.

    (( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.) / n

    Parameters
    ----------
    * u (``npt.NDArray``): Coordenadas do primeiro ponto.
    * v (``npt.NDArray``): Coordenadas do segundo ponto.
    * p (``float``): O parâmetro p define o tipo de distância a ser calculada:
        - p = 1: Distância **Manhattan** — soma das diferenças absolutas.
        - p = 2: Distância **Euclidiana** — soma das diferenças ao quadrado (raiz quadrada).
        - p > 2: Distância **Minkowski** com uma penalidade crescente à medida que p aumenta.

    returns
    ----------
    * Distância (``float``) entre os dois pontos.

    """
    return (np.sum(np.abs(u - v) ** p) ** (1/p)) / len(u)

@njit
def compute_metric_distance(
    u: npt.NDArray,
    v: npt.NDArray,
    metric="euclidean",
    p=2
):
    """
    Function to calculate the distance between two points by the chosen ``metric``.

    Parameters
    ----------
    * u (``npt.NDArray``): Coordinates of the first point.
    * v (``npt.NDArray``): Coordinates of the second point.

    returns
    ----------
    * Distance (``double``) between the two points with the selected metric.

    ----

    Função para calcular a distância entre dois pontos pela ``métrica`` escolhida.

    Parameters
    ----------
    * u (``npt.NDArray``): Coordenadas do primeiro ponto.
    * v (``npt.NDArray``): Coordenadas do segundo ponto.

    returns
    ----------
    * Distância (``double``) entre os dois pontos com a métrica selecionada.

    """
    if metric == "hamming":
        return hamming(u, v)
    if metric == "cityblock":
        return cityblock(u, v)
    if metric == "minkowski":
        return minkowski(u, v, p)

    return euclidean(u, v)

@njit()
def min_distance_to_class_vectors(
    x_class: npt.NDArray,
    vector_x: npt.NDArray,
    metric: str,
    p: int
) -> float:
    """
    Calculates the minimum distance between an input vector and the vectors of a class.

    Parameters
    ----------
    * x_class (``npt.NDArray``): Array containing the class vectors to be compared 
    with the input vector. Expected shape: (n_samples, n_features).
    * vector_x (``npt.NDArray``): Vector to be compared with the class vectors.
    Expected shape: (n_features,).
    * metric (``str``): Distance metric to be used. Available options: 
    ["hamming", "cityblock", "minkowski", "euclidean"]
    * p (``float``): Parameter for the Minkowski distance (used only if `metric` 
    is "minkowski").

    Returns
    ----------
    * float: The minimum distance calculated between the input vector and the class vectors.
    * Returns -1.0 if the input dimensions are incompatible.

    --------
    
    Calcula a menor distância entre um vetor de entrada e os vetores de uma classe.
    
    Parameters
    ----------
    * x_class (``npt.NDArray``): Array contendo os vetores da classe com os quais o vetor de
        entrada será comparado. Formato esperado: (n_amostras, n_características).
    * vector_x (``npt.NDArray``): Vetor a ser comparado com os vetores da classe.
        Formato esperado: (n_características,).
    * metric (``str``): Métrica de distância a ser utilizada. Opções disponíveis: 
        ["hamming", "cityblock", "minkowski", "euclidean"]
    * p (``float``): Parâmetro da métrica de Minkowski (utilizado apenas se `metric` for 
        "minkowski").
    
    Returns
    ----------
    * float: A menor distância calculada entre o vetor de entrada e os vetores da classe.
    * Retorna -1.0 se as dimensões de entrada forem incompatíveis.
    """
    n = x_class.shape[1]
    if n != vector_x.shape[0]:
        return -1.0

    min_distance = np.inf
    for i in range(x_class.shape[0]):
        distance = compute_metric_distance(vector_x, x_class[i], metric, p)
        min_distance = min(min_distance, distance)

    return min_distance
