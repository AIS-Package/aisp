"""
The functions perform utilize Numba decorators for Just-In-Time compilation.

Contains functions that generate sets of mutated clones from continuous or binary vectors,
simulating the clonal expansion process in artificial immune systems.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, types


@njit([(types.float64[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_continuous(
    vector: npt.NDArray[np.float64],
    n: int,
    mutation_rate: float
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
    mutation_rate : float, default=1
        If 0 <= mutation_rate < 1: probability of mutating each component.
        If mutation_rate >= 1 or mutation_rate <= 0: the mutation randomizes
        a number of components between 1 and len(vector).

    Returns
    -------
    clone_set : npt.NDArray
        An Array(n, len(vector)) containing the `n` mutated clones of the original vector.
    """
    n_features = vector.shape[0]
    clone_set = np.empty((n, n_features), dtype=np.float64)
    for i in range(n):
        clone = vector.copy()
        if 0 <= mutation_rate < 1:
            for j in range(n_features):
                if np.random.random() < mutation_rate:
                    clone[j] = np.random.random()
        else:
            n_mutations = np.random.randint(1, n_features + 1)
            position_mutations = np.random.permutation(n_features)[:n_mutations]
            for j in range(n_mutations):
                idx = position_mutations[j]
                clone[idx] = np.float64(np.random.random())
        clone_set[i] = clone

    return clone_set


@njit([(types.boolean[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_binary(
    vector: npt.NDArray[np.bool_],
    n: int,
    mutation_rate: float
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
    mutation_rate : float, default=1
        If 0 <= mutation_rate < 1: probability of mutating each component.
        If mutation_rate >= 1 or mutation_rate <= 0: the mutation randomizes
        a number of components between 1 and len(vector).

    Returns
    -------
    clone_set : npt.NDArray[np.bool_]
        An Array(n, len(vector)) containing the `n` mutated clones of the original vector.
    """
    n_features = vector.shape[0]
    clone_set = np.empty((n, n_features), dtype=np.bool_)
    for i in range(n):
        clone = vector.copy()
        if 0 <= mutation_rate < 1:
            for j in range(n_features):
                if np.random.random() < mutation_rate:
                    clone[j] = not clone[j]
        else:
            n_mutations = np.random.randint(1, n_features + 1)
            position_mutations = np.random.permutation(n_features)[:n_mutations]
            for j in range(n_mutations):
                idx = position_mutations[j]
                clone[idx] = not clone[idx]
        clone_set[i] = clone

    return clone_set


@njit([(types.float64[:], types.int64, types.float64[:, :], types.float64)], cache=True)
def clone_and_mutate_ranged(
    vector: npt.NDArray[np.float64],
    n: int,
    bounds: npt.NDArray[np.float64],
    mutation_rate: float,
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
    mutation_rate : float, default=1
        If 0 <= mutation_rate < 1: probability of mutating each component.
        If mutation_rate >= 1 or mutation_rate <= 0: the mutation randomizes
        a number of components between 1 and len(vector).

    Returns
    -------
    clone_set : npt.NDArray
        An Array(n, len(vector)) containing the `n` mutated clones of the original vector.
    """
    n_features = vector.shape[0]
    clone_set = np.empty((n, n_features), dtype=np.float64)

    for i in range(n):
        clone = vector.copy()
        if 0 <= mutation_rate < 1:
            for j in range(n_features):
                if np.random.random() < mutation_rate:
                    clone[j] = np.random.uniform(low=bounds[0][j], high=bounds[1][j])
        else:
            n_mutations = np.random.randint(1, n_features + 1)
            position_mutations = np.random.permutation(n_features)[:n_mutations]
            for j in range(n_mutations):
                idx = position_mutations[j]
                min_limit = bounds[0][idx]
                max_limit = bounds[1][idx]
                clone[idx] = np.random.uniform(low=min_limit, high=max_limit)
        clone_set[i] = clone

    return clone_set


@njit([(types.int64[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_permutation(
    vector: npt.NDArray[np.int64],
    n: int,
    mutation_rate: float
) -> npt.NDArray[np.int64]:
    """Generate a set of mutated clones by random permutation.

    Parameters
    ----------
    vector : npt.NDArray[np.int64]
        The original immune cell with permutation values to be cloned and mutated.
    n : int
        The number of mutated clones to be generated.
    mutation_rate : float
        Probability of mutating each component 0 <= mutation_rate < 1.

    Returns
    -------
    clone_set : npt.NDArray
        An Array(n, len(vector)) containing the `n` mutated clones of the original vector.
    """
    n_features = vector.shape[0]
    clone_set = np.empty((n, n_features), dtype=np.int64)

    for i in range(n):
        clone = vector.copy()
        for j in range(n_features):
            if np.random.random() < mutation_rate:
                idx = np.random.randint(0, n_features)
                tmp = clone[j]
                clone[j] = clone[idx]
                clone[idx] = tmp
        clone_set[i] = clone

    return clone_set
