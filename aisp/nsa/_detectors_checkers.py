"""
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
