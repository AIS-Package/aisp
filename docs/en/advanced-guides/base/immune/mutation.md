# Mutation

Contains functions that generate sets of mutated clones from continuous or binary vectors, simulating the clonal expansion process in artificial immune systems.

## clone_and_mutate_continuous

```python
@njit([(types.float64[:], types.int64)], cache=True)
def clone_and_mutate_continuous(
    vector: npt.NDArray[np.float64],
    n: int,
    mutation_rate: float
) -> npt.NDArray[np.float64]:
```

Generates a set of mutated clones from a continuous vector.

This function creates `n` clones of the input vector and applies random mutations to each one, simulating the clonal expansion process in artificial immune systems. Each clone receives a random number of mutations at distinct positions of the original vector.

### Parameters

* `vector` (`npt.NDArray[np.float64]`): The original immune cell with continuous values to be cloned and mutated.
* `n` (`int`): Number of mutated clones to be generated.
* ``mutation_rate`` : (``float``)  If 0 <= mutation_rate < 1: probability of mutating each component.
  If mutation_rate >= 1 or mutation_rate <= 0: the mutation randomizes
  number of components between 1 and len(vector).

### Returns

* `clone_set` (`npt.NDArray[np.float64]`): Array with shape `(n, len(vector))` containing the `n` mutated clones of the original vector.

---

## clone_and_mutate_binary

```python
@njit([(types.boolean[:], types.int64)], cache=True)
def clone_and_mutate_binary(
    vector: npt.NDArray[np.bool_],
    n: int
) -> npt.NDArray[np.bool_]:
```

Generates a set of mutated clones from a binary vector.

This function creates `n` clones of the input binary vector and applies random mutations to some bits, simulating clonal expansion in artificial immune systems with discrete representations.

### Parameters

* `vector` (`npt.NDArray[np.bool_]`): The original immune cell with binary values to be cloned and mutated.
* `n` (`int`): Number of mutated clones to be generated.

### Returns

* `clone_set` (`npt.NDArray[np.bool_]`): Array with shape `(n, len(vector))` containing the `n` mutated clones of the original vector.

---

## clone_and_mutate_ranged

```python
@njit([(types.float64[:], types.int64, types.float64[:, :])], cache=True)
def clone_and_mutate_ranged(
    vector: npt.NDArray[np.float64],
    n: int,
    bounds: npt.NDArray[np.float64],
    mutation_rate: float
) -> npt.NDArray[np.float64]:
```

Generates a set of mutated clones from a continuous vector using custom bounds per dimension.

This function creates `n` clones of the input vector and applies random mutations to each of them, simulating the process of clonal expansion in artificial immune systems. Each clone will have a random number of mutations applied to distinct positions of the original vector, respecting the mutation bounds defined per dimension.

### Parameters

* `vector` (`npt.NDArray[np.float64]`): The original immune cell with continuous values to be cloned and mutated.
* `n` (`int`): Number of mutated clones to be generated.
* `bounds` (`npt.NDArray[np.float64]`): A 2D array with shape `(len(vector), 2)` containing the minimum and maximum values for each dimension.
* ``mutation_rate`` : (``float``)  If 0 <= mutation_rate < 1: probability of mutating each component.
  If mutation_rate >= 1 or mutation_rate <= 0: the mutation randomizes
  number of components between 1 and len(vector).

### Returns

* `clone_set` (`npt.NDArray[np.float64]`): Array with shape `(n, len(vector))` containing the `n` mutated clones of the original vector.

---

## clone_and_mutate_permutation

```python
@njit([(types.int64[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_permutation(
    vector: npt.NDArray[np.int64],
    n: int,
    mutation_rate: float
) -> npt.NDArray[np.int64]:
```

Generates a set of mutated clones from a permutation vector.

This function creates `n` clones of the input permutation vector and applies random mutations to each one, simulating clonal expansion in artificial immune systems with discrete permutations. Each clone receives a random number of swaps according to the mutation rate.

### Parameters

* `vector` (`npt.NDArray[np.int64]`): The original immune cell with permutation values to be cloned and mutated.
* `n` (`int`): Number of mutated clones to be generated.
* `mutation_rate` (`float`): Probability of mutating each component (0 <= mutation_rate < 1).

### Returns

* `clone_set` (`npt.NDArray[np.int64]`): Array with shape `(n, len(vector))` containing the `n` mutated clones of the original vector.

---
