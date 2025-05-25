# Mutation

Contains functions that generate sets of mutated clones from continuous or binary vectors, simulating the clonal expansion process in artificial immune systems.

## clone_and_mutate_continuous

```python
@njit([(types.float64[:], types.int64)], cache=True)
def clone_and_mutate_continuous(
    vector: npt.NDArray[np.float64],
    n: int
) -> npt.NDArray[np.float64]:
```

Generates a set of mutated clones from a continuous vector.

This function creates `n` clones of the input vector and applies random mutations to each one, simulating the clonal expansion process in artificial immune systems. Each clone receives a random number of mutations at distinct positions of the original vector.

### Parameters

* `vector` (`npt.NDArray[np.float64]`): The original immune cell with continuous values to be cloned and mutated.
* `n` (`int`): Number of mutated clones to be generated.

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