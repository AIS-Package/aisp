"""ns: Negative Selection

The functions perform detector checks and utilize Numba decorators for Just-In-Time compilation
"""

import numpy as np
import numpy.typing as npt
from numba import njit


@njit()
def check_detector_bnsa_validity(
    x_class: npt.NDArray, vector_x: npt.NDArray, aff_thresh: float
) -> bool:
    """
    Checks the validity of a candidate detector (vector_x) against samples from a class (x_class)
    using the Hamming distance. A detector is considered INVALID if its distance to any sample
    in ``x_class`` is less than or equal to ``aff_thresh``.

    Parameters
    ----------
    * x_class (``NDArray``): 2D NumPy array containing the class samples.
    * vector_x (``NDArray``): 1D NumPy array representing the detector.
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
    * x_class (``NDArray``): Array NumPy 2D contendo as amostras da classe.
    * vector_x (): Array NumPy 1D representando o detector.
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
        distance = np.sum(vector_x != x_class[i]) / n
        if distance <= aff_thresh:
            return False
    return True


@njit()
def bnsa_class_prediction(
    features: np.ndarray,
    class_detectors: np.ndarray,
    aff_thresh: float
) -> int:
    """
    Define a classe de uma amostra a partir dos detectores não-próprios.

    Parameters
    ----------
    * features (np.ndarray): amostra binária a ser classificada (shape: [n_features]).
    *  class_detectors (np.ndarray): Matriz 3D contendo os detectores de todas as classes 
        (shape: [n_classes, n_detectors, n_features]).
    * aff_thresh (float): Limiar de afinidade que determina se um detector reconhece a amostra como
        não-própria.

    Returns
    ----------
    * int: Índice da classe predita. Retorna -1 se for não-própria para todas as classes.
    """
    n_classes, n_detectors, n_features = class_detectors.shape
    best_class_idx = -1
    best_avg_distance = 0

    for class_index in range(n_classes):
        total_distance = 0.0
        class_found = True

        # Calculates the Hamming distance between the row and all detectors.
        for detector_index  in range(n_detectors):
             # Calcula a distância de Hamming normalizada entre a amostra e o detector
            distance = np.sum(
                features != class_detectors[class_index][detector_index]
            ) / n_features

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
