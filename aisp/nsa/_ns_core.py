"""ns: Negative Selection

The functions perform detector checks and utilize Numba decorators for Just-In-Time compilation
"""

import numpy.typing as npt
from numba import njit

from aisp.utils.distance import compute_metric_distance, hamming


@njit(cache=True)
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

    ----

    Verifica a validade de um candidato a detector (vector_x) contra amostras de uma classe
    (x_class) usando a distância de Hamming. Um detector é considerado INVÁLIDO se a sua distância
    para qualquer amostra em ``x_class`` for menor ou igual a ``aff_thresh``.

    Parameters
    ----------
    * x_class (``npt.NDArray``): Array contendo as amostras da classe. Formato esperado:
        (n_amostras, n_características).
    * vector_x (``npt.NDArray``): Array representando o detector. Formato esperado:
        (n_características,).
    * aff_thresh (``float``): Limiar de afinidade.

    Returns
    ----------
    * True se o detector for válido, False caso contrário.
    """
    n = x_class.shape[1]
    if n != vector_x.shape[0]:
        return False

    for i in range(x_class.shape[0]):
        # Calculate the normalized Hamming Distance
        if hamming(x_class[i], vector_x) <= aff_thresh:
            return False
    return True


@njit(cache=True)
def bnsa_class_prediction(
    features: npt.NDArray,
    class_detectors: npt.NDArray,
    aff_thresh: float
) -> int:
    """
    Define a classe de uma amostra a partir dos detectores não-próprios.

    Parameters
    ----------
    * features (``npt.NDArray``): amostra binária a ser classificada (shape: [n_features]).
    *  class_detectors (``npt.NDArray``): Matriz contendo os detectores de todas as classes 
        (shape: [n_classes, n_detectors, n_features]).
    * aff_thresh (``float``): Limiar de afinidade que determina se um detector reconhece a
        amostra como não-própria.

    Returns
    ----------
    * int: Índice da classe predita. Retorna -1 se for não-própria para todas as classes.
    """
    n_classes, n_detectors, _ = class_detectors.shape
    best_class_idx = -1
    best_avg_distance = 0

    for class_index in range(n_classes):
        total_distance = 0.0
        class_found = True

        # Calculates the Hamming distance between the row and all detectors.
        for detector_index  in range(n_detectors):
            # Calcula a distância de Hamming normalizada entre a amostra e o detector
            distance = hamming(features, class_detectors[class_index][detector_index])

            # Se a distância for menor ou igual ao limiar, o detector reconhece a amostra
            # como não-própria
            if distance <= aff_thresh:
                class_found = False
                break
            total_distance += distance

        # se a amostrar é própria para a classe
        if class_found:
            avg_distance = total_distance / n_detectors
            # Escolhe a classe com a maior distância média.
            if avg_distance > best_avg_distance:
                best_avg_distance = avg_distance
                best_class_idx = class_index

    return best_class_idx


@njit(cache=True)
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
    * aff_thresh (``float``): Affinity threshold.

    Returns
    ----------
    * True if the detector is valid, False otherwise.

    ----

    Verifica a validade de um candidato a detector (vector_x) contra amostras de uma classe
    (x_class) usando a distância de Hamming. Um detector é considerado INVÁLIDO se a sua distância
    para qualquer amostra em ``x_class`` for menor ou igual a ``aff_thresh``.

    Parameters
    ----------
    * x_class (``npt.NDArray``): Array contendo as amostras da classe. Formato esperado:
        (n_amostras, n_características).
    * vector_x (``npt.NDArray``): Array representando o detector. Formato esperado:
        (n_características,).
    * aff_thresh (``float``): Limiar de afinidade.

    Returns
    ----------
    * True se o detector for válido, False caso contrário.
    """
    n = x_class.shape[1]
    if n != vector_x.shape[0]:
        return False

    for i in range(x_class.shape[0]):
        distance = compute_metric_distance(vector_x, x_class[i], metric, p)
        if distance <= threshold:
            return False
    return True
