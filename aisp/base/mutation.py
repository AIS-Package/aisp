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
    Generates a set of mutated clones from a cell represented by a continuous vector.

    This function creates `n` clones of the input vector and applies random mutations to each of
    them, simulating the process of clonal expansion in artificial immune systems. Each clone
    will have a random number of mutations applied in distinct positions of the original vector.

    Parameters
    ----------
    * vector (``ndarray``): The original immune cell with continuous values to be cloned and
        mutated.
    * n (``int``): The number of mutated clones to be generated.

    Returns
    ----------
    * ``ndarray``: An Array(n, len(vector)) containing the `n` mutated clones of the original
        vector.
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
    Generates a set of mutated clones from a cell represented by a binary vector.

    This function creates `n` clones of the input vector and applies random mutations to each of
    them, changing some bits randomly. The process simulates clonal expansion in artificial
    immune systems with discrete representations.

    Parameters
    ----------
    * vector (``ndarray``): The original immune cell with binary values to be cloned and mutated.
    * n (``int``): The number of mutated clones to be generated.

    Returns
    ----------
    * ``ndarray``: An Array(n, len(vector)) containing the `n` mutated clones of the original
        vector.
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
