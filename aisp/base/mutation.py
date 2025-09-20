"""
The functions perform utilize Numba decorators for Just-In-Time compilation.

Contains functions that generate sets of mutated clones from continuous or binary vectors,
simulating the clonal expansion process in artificial immune systems.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, types


@njit([(types.float64[:], types.int64)], cache=True)
def clone_and_mutate_continuous(
    vector: npt.NDArray[np.float64],
    n: int
) -> npt.NDArray[np.float64]:
    """
    Generate a set of mutated clones from a cell represented by a continuous vector.

    This function creates `n` clones of the input vector and applies random mutations to each of
    them, simulating the process of clonal expansion in artificial immune systems. Each clone
    will have a random number of mutations applied in distinct positions of the original vector.

    Parameters
    ----------
    vector : npt.NDArray[np.float64]
        The original immune cell with continuous values to be cloned and mutated.
    n : int
        The number of mutated clones to be generated.

    Returns
    -------
    clone_set : npt.NDArray
        An Array(n, len(vector)) containing the `n` mutated clones of the original vector.
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
    Generate a set of mutated clones from a cell represented by a binary vector.

    This function creates `n` clones of the input vector and applies random mutations to each of
    them, changing some bits randomly. The process simulates clonal expansion in artificial
    immune systems with discrete representations.

    Parameters
    ----------
    vector : npt.NDArray[np.bool_]
        The original immune cell with binary values to be cloned and mutated.
    n : int
        The number of mutated clones to be generated.

    Returns
    -------
    clone_set : npt.NDArray[np.bool_]
        An Array(n, len(vector)) containing the `n` mutated clones of the original vector.
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


@njit([(types.float64[:], types.int64, types.float64[:, :])], cache=True)
def clone_and_mutate_ranged(
    vector: npt.NDArray[np.float64],
    n: int,
    bounds: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Generate a set of mutated clones from a cell represented by custom ranges per dimension.

    This function creates `n` clones of the input vector and applies random mutations to each of
    them, simulating the process of clonal expansion in artificial immune systems. Each clone
    will have a random number of mutations applied in distinct positions of the original vector.

    Parameters
    ----------
    vector : npt.NDArray[np.bool_]
        The original immune cell with binary values to be cloned and mutated.
    n : int
        The number of mutated clones to be generated.
    bounds : np.ndarray
        Array (n_features, 2) with min and max per dimension.

    Returns
    -------
    clone_set : npt.NDArray
        An Array(n, len(vector)) containing the `n` mutated clones of the original vector.
    """
    n_features = vector.shape[0]
    clone_set = np.empty((n, n_features), dtype=np.float64)

    for i in range(n):
        n_mutations = np.random.randint(1, n_features)
        clone = vector.copy()
        position_mutations = np.random.permutation(n_features)[:n_mutations]
        for j in range(n_mutations):
            idx = position_mutations[j]
            min_limit = bounds[0][idx]
            max_limit = bounds[1][idx]
            clone[idx] = np.random.uniform(min_limit, max_limit)
        clone_set[i] = clone

    return clone_set
