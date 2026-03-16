---
id: mutation
sidebar_label: mutation
keywords:
    - mutation
    - clonal expansion
    - immune system
    - python numba functions
    - vector mutation
---

# mutation

The functions perform utilize Numba decorators for Just-In-Time compilation.

Contains functions that generate sets of mutated clones from continuous or binary vectors,
simulating the clonal expansion process in artificial immune systems.

> **Module:** `aisp.base.immune`  
> **Import:** `from aisp.base.immune import mutation`

## Functions

### clone_and_mutate_continuous

```python
@njit([(types.float64[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_continuous(
    vector: npt.NDArray[np.float64],
    n: int,
    mutation_rate: float
) -> npt.NDArray[np.float64]:
    ...
```

Generate a set of mutated clones from a cell represented by a continuous vector.

This function creates `n` clones of the input vector and applies mutations to each of
them, simulating the process of clonal expansion in artificial immune systems.

**Parameters**

| Name            | Type                      | Default | Description                                                                                                      |
|-----------------|---------------------------|:-------:|------------------------------------------------------------------------------------------------------------------|
| `vector`        | `npt.NDArray[np.float64]` |    -    | The original immune cell with continuous values to be cloned and mutated.                                        |
| `n`             | `int`                     |    -    | The number of mutated clones to be generated.                                                                    |
| `mutation_rate` | `float`                   |    -    | If `0 ≤ mutation_rate < 1`, mutation probability per feature. Otherwise, a random number of features is mutated. |

**Returns**

`npt.NDArray[np.float64]` - An Array(n, len(vector)) containing the `n` mutated clones of the original vector.

### clone_and_mutate_binary

```python
@njit([(types.boolean[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_binary(
    vector: npt.NDArray[np.bool_],
    n: int,
    mutation_rate: float
) -> npt.NDArray[np.bool_]:
    ...
```

Generate a set of mutated clones from a cell represented by a binary vector.

This function creates `n` clones of the input binary vector and applies mutations to the
bits, simulating clonal expansion in artificial immune systems with discrete representations.

**Parameters**

| Name            | Type                      | Default | Description                                                                                                      |
|-----------------|---------------------------|:-------:|------------------------------------------------------------------------------------------------------------------|
| `vector`        | `npt.NDArray[np.float64]` |    -    | The original immune cell with binary values to be cloned and mutated.                                            |
| `n`             | `int`                     |    -    | The number of mutated clones to be generated.                                                                    |
| `mutation_rate` | `float`                   |    -    | If `0 ≤ mutation_rate < 1`, mutation probability per feature. Otherwise, a random number of features is mutated. |

**Returns**

`npt.NDArray[np.bool_]` - An Array(n, len(vector)) containing the `n` mutated clones of the original vector.

### clone_and_mutate_ranged

```python
@njit([(types.float64[:], types.int64, types.float64[:, :], types.float64)], cache=True)
def clone_and_mutate_ranged(
    vector: npt.NDArray[np.float64],
    n: int,
    bounds: npt.NDArray[np.float64],
    mutation_rate: float,
) -> npt.NDArray[np.float64]:
    ...
```

Generate a set of mutated clones from a cell represented by custom ranges per dimension.

This function creates `n` clones of the input vector and applies mutations to each of
them, simulating the process of clonal expansion in artificial immune systems.

**Parameters**

| Name            | Type                      | Default | Description                                                                                                      |
|-----------------|---------------------------|:-------:|------------------------------------------------------------------------------------------------------------------|
| `vector`        | `npt.NDArray[np.float64]` |    -    | The original immune cell with continuous values to be cloned and mutated.                                        |
| `n`             | `int`                     |    -    | The number of mutated clones to be generated.                                                                    |
| `bounds`        | `np.ndarray`              |    -    | Array (n_features, 2) with min and max per dimension.                                                            |
| `mutation_rate` | `float`                   |    -    | If `0 ≤ mutation_rate < 1`, mutation probability per feature. Otherwise, a random number of features is mutated. |

**Returns**

`npt.NDArray[np.float64]` - An Array(n, len(vector)) containing the `n` mutated clones of the original vector.

### clone_and_mutate_continuous

```python
@njit([(types.int64[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_permutation(
    vector: npt.NDArray[np.int64],
    n: int,
    mutation_rate: float
) -> npt.NDArray[np.int64]:
    ...
```

Generate a set of mutated clones by permutation.

**Parameters**

| Name            | Type                    | Default | Description                                                                |
|-----------------|-------------------------|:-------:|----------------------------------------------------------------------------|
| `vector`        | `npt.NDArray[np.int64]` |    -    | The original immune cell with permutation values to be cloned and mutated. |
| `n`             | `int`                   |    -    | The number of mutated clones to be generated.                              |
| `mutation_rate` | `float`                 |    -    | Probability of mutating each feature 0 ≤ mutation_rate < 1.                |

**Returns**

`npt.NDArray[np.float64]` - An Array(n, len(vector)) containing the `n` mutated clones of the original vector.
