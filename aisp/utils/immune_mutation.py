"""airs: Artificial Immune Recognition System

The functions perform utilize Numba decorators for Just-In-Time compilation
"""

import numpy as np
import numpy.typing as npt
from numba import njit


@njit()
def generate_mutated_clones(
    vector: npt.NDArray,
    algorithm: str,
    n: int
):
    """
    Gera um conjunto de clones mutados a partir de uma célula (``vetor``).

    Esta função cria `n` clones do vector de entrada e aplica a mutação
    a cada clones Ela simula os processos de expansão clonal.

    Parâmetros
    ----------
    * vector (``ndarray``): A célula imune original a ser clonada e mutada.
    * algorithm (``str``): Dicionário contendo parâmetros de mutação, como taxa de mutação,
        tipo de mutação ou fatores dependentes de afinidade.
    * n (``int``): O número de clones mutados a serem gerados.

    Retorna
    ----------
    * ``ndarray``: Um array de `n` clones mutados derivados do vetor original.
    """
    n_features = vector.shape[0]
    clone_set = np.empty((n, n_features), dtype=np.float64)
    i = 0
    for i in range(n):
        n_mutations = np.random.randint(1, n_features)
        clone = vector.copy()
        position_mutations = np.random.choice(
            np.arange(n_features), size=n_mutations, replace=False
        )
        for j in range(n_mutations):
            idx = position_mutations[j]
            if algorithm == "binary-features":
                clone[idx] = np.bool_(np.random.randint(0, 2))
            else:
                clone[idx] = np.random.random()
        clone_set[i] = clone

    return clone_set
