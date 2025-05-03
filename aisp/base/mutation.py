"""airs: Artificial Immune Recognition System

The functions perform utilize Numba decorators for Just-In-Time compilation
"""

import numpy as np
import numpy.typing as npt
from numba import njit, types


@njit([(types.float64[:], types.int64)], cache=True)
def clone_and_mutate_continuous(
    vector: npt.NDArray,
    n: int
) -> npt.NDArray[np.float64]:
    """
    Gera um conjunto de clones mutados a partir de uma célula (``vetor``).

    Esta função cria `n` clones do vector de entrada e aplica a mutação
    a cada clones Ela simula os processos de expansão clonal.

    Parâmetros
    ----------
    * vector (``ndarray``): A célula imune original a ser clonada e mutada.
    * n (``int``): O número de clones mutados a serem gerados.

    Retorna
    ----------
    * ``ndarray``: Um array de `n` clones mutados derivados do vetor original.
    """
    n_features = vector.shape[0]
    clone_set = np.empty((n, n_features), dtype=np.float64)
    for i in range(n):
        n_mutations = np.random.randint(1, n_features)
        clone = vector.copy()
        position_mutations = np.random.permutation(n_features)[:n_mutations]
        for j in range(n_mutations):
            idx = position_mutations[j]
            clone[idx] = np.float64(np.random.random())
        clone_set[i] = clone

    return clone_set


@njit([(types.boolean[:], types.int64)], cache=True)
def clone_and_mutate_binary(
    vector: npt.NDArray[np.bool_],
    n: int
) -> npt.NDArray[np.bool_]:
    """
    Gera um conjunto de clones mutados a partir de uma célula (``vetor``).

    Esta função cria `n` clones do vector de entrada e aplica a mutação
    a cada clones Ela simula os processos de expansão clonal.

    Parâmetros
    ----------
    * vector (``ndarray``): A célula imune original a ser clonada e mutada.
    * n (``int``): O número de clones mutados a serem gerados.

    Retorna
    ----------
    * ``ndarray``: Um array de `n` clones mutados derivados do vetor original.
    """
    n_features = vector.shape[0]
    clone_set = np.empty((n, n_features), dtype=np.bool_)
    for i in range(n):
        n_mutations = np.random.randint(1, n_features)
        clone = vector.copy()
        position_mutations = np.random.permutation(n_features)[:n_mutations]
        for j in range(n_mutations):
            idx = position_mutations[j]
            clone[idx] = np.bool_(np.random.randint(0, 2))
        clone_set[i] = clone

    return clone_set
