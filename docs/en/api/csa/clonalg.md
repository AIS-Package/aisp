---
id: clonalg
sidebar_label: Clonalg
keywords:
    - optimization
    - clonal selection
    - clonalg
    - antibody population
    - objective function
tags:
    - optimization
    - clonal selection
    - minimization
    - maximization
    - binary
    - continuous
    - permutation
    - ranged
---

# Clonalg

Clonal Selection Algorithm (CLONALG).

:::tip[Inheritance]

This class extends [BaseOptimizer](../base/base-optimizer.md)

:::


> **Module:** `aisp.csa`  
> **Import:** `from aisp.csa import Clonalg`

---

## Overview

The Clonal Selection Algorithm (CSA) is an optimization algorithm inspired by the biological
process of clonal selection and expansion of antibodies in the immune system [^1]. This
implementation of CLONALG has been adapted for the minimization or maximization of cost
functions in binary, continuous, ranged-value, and permutation problems.

:::note

This CLONALG implementation contains some changes based on the AISP context, for general
application to various problems, which may produce results different from the standard or
specific implementation. This adaptation aims to generalize CLONALG to minimization and
maximization tasks, in addition to supporting continuous, discrete, and permutation problems.

:::

---

## Example

```python
import numpy as np
from aisp.csa import Clonalg
# Search space limits
bounds = {'low': -5.12, 'high': 5.12}
# Objective function
def rastrigin_fitness(x):
    x = np.clip(x, bounds['low'], bounds['high'])
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
# CLONALG Instance
clonalg = Clonalg(problem_size=2, bounds=bounds, seed=1)
clonalg.register('affinity_function', rastrigin_fitness)
population = clonalg.optimize(100, 50, False)
print('Best cost:', abs(clonalg.best_cost))
```
**Output:**
```bash
Best cost: 0.02623036956750724
```

---

## Constructor Parameters

| Name                    | Type                                   |       Default       | Description                                                                                                                                                |
|-------------------------|----------------------------------------|:-------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `problem_size`          | `int`                                  |          -          | Dimension of the problem to be minimized.                                                                                                                  |
| `N`                     | `int`                                  |        `50`         | Number of memory cells (antibodies) in the population.                                                                                                     |
| `rate_clonal`           | `int`                                  |        `10`         | Maximum number of possible clones of a cell. This value is multiplied by cell_affinity to determine the number of clones.                                  |
| `rate_hypermutation`    | `float`                                |        `1.0`        | Hypermutation rate controls the intensity of mutations during clonal expansion. Higher values decrease mutation intensity, while lower values increase it. |
| `n_diversity_injection` | `int`                                  |         `5`         | Number of new random memory cells injected to maintain diversity.                                                                                          |
| `selection_size`        | `int`                                  |         `5`         | Number of the best antibodies selected for cloning.                                                                                                        |
| `affinity_function`     | `Optional[Callable[..., npt.NDArray]]` |       `None`        | Objective function to evaluate candidate solutions in minimizing the problem.                                                                              |
| `feature_type`          | [`FeatureTypeAll`](../utils/types.md)  | `'ranged-features'` | Type of problem samples: binary, continuous, or based on value ranges.                                                                                     |
| `bounds`                | `Optional[Dict]`                       |       `None`        | Definition of search limits when ``feature_type='ranged-features'``.                                                                                       |
| `mode`                  | `{"min", "max"}`                       |       `'min'`       | Defines whether the algorithm minimizes or maximizes the cost function.                                                                                    |
| `seed`                  | `int`                                  |       `None`        | Seed for random generation of detector values. If None, the value is random.                                                                               |

## Attributes

| Name         | Type                       | Default | Description              |
|--------------|----------------------------|:-------:|--------------------------|
| `population` | `Optional[List[Antibody]]` | `None`  | Population of antibodies |

---

## Public Methods

### optimize

```python
def optimize(
    self, max_iters: int = 50, n_iter_no_change=10, verbose: bool = True
) -> List[Antibody]:
    ...
```

Execute the optimization process and return the population.

**Parameters**

| Name               | Type   | Default | Description                                                                      |
|--------------------|--------|:-------:|----------------------------------------------------------------------------------|
| `max_iters`        | `int`  |  `50`   | Maximum number of iterations when searching for the best solution using clonalg. |
| `n_iter_no_change` | `int`  |  `10`   | The maximum number of iterations without updating the best cell.                 |
| `verbose`          | `bool` | `True`  | Feedback on iterations, indicating the best antibody.                            |

**Raises**

* `NotImplementedError` - If no affinity function has been provided to model.

**Returns**
 
**population** : `List[Antibody]` - Antibody population after clonal expansion.

---

### affinity_function

```python
def affinity_function(self, solution: npt.NDArray) -> np.float64:
    ...
```

Evaluate the affinity of a candidate cell.

**Parameters**

| Name       | Type          | Default | Description                     |
|------------|---------------|:-------:|---------------------------------|
| `solution` | `npt.NDArray` |    -    | Candidate solution to evaluate. |

**Raises**

* `NotImplementedError` - If no affinity function has been provided.

**Returns**

**affinity** : `np.float64` - Affinity value associated with the given cell.

---

## Extended Example

Complete usage examples are available in the Jupyter Notebooks:

- [**Knapsack Problem Example**](../../../../examples/en/optimization/clonalg/knapsack_problem_example.ipynb)
- [**Rastrigin Function Example**](../../../../examples/en/optimization/clonalg/rastrigin_function_example.ipynb)
- [**Tsp Problem Example**](../../../../examples/en/optimization/clonalg/tsp_problem_example.ipynb)

---

## References

[^1]: BROWNLEE, Jason. Clonal Selection Algorithm. Clever Algorithms: Nature-inspired 
    Programming Recipes., 2011. Available at:
    https://cleveralgorithms.com/nature-inspired/immune/clonal_selection_algorithm.html
