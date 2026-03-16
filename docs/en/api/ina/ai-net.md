---
id: ai-net
sidebar_label: AiNet
keywords:
    - immune network
    - clustering
    - data compression
    - unsupervised learning
    - Minimum Spanning Tree
tags:
    - clustering
    - unsupervised
    - immune network
    - data compression
---

# AiNet

Artificial Immune Network (AiNet) for Compression and Clustering.

:::tip[Inheritance]

This class extends [BaseClusterer](../base/base-clusterer.md).

:::


> **Module:** `aisp.ina`  
> **Import:** `from aisp.ina import AiNet`

---

## Overview

This class implements the aiNet algorithm, an artificial immune network model designed for
clustering and data compression tasks. The aiNet algorithm uses principles from immune
network theory, clonal selection, and affinity maturation to compress high-dimensional
datasets [^1].  
For clustering, the class uses SciPy implementation of the **Minimum Spanning Tree**
(MST) to remove the most distant nodes and separate the groups [^2].

---

## Example

```python
import numpy as np
from aisp.ina import AiNet

np.random.seed(1)
# Generating training data
a = np.random.uniform(high=0.4, size=(50, 2))
b = np.random.uniform(low=0.6, size=(50, 2))
x_train = np.vstack((a, b))
# AiNet Instance
ai_net = AiNet(
    N=150,
    mst_inconsistency_factor=1,
    seed=1,
    affinity_threshold=0.85,
    suppression_threshold=0.7
)
ai_net = ai_net.fit(x_train, verbose=True)
x_test = [
    [0.15, 0.45],  # Expected: label 0
    [0.85, 0.65],  # Esperado: label 1
]
y_pred = ai_net.predict(x_test)
print(y_pred)
```
**Output**
```bash
[0 1]
```

---

## Constructor Parameters

| Name                       | Type                                         |    Default    | Description                                                                                                                                |
|----------------------------|----------------------------------------------|:-------------:|--------------------------------------------------------------------------------------------------------------------------------------------|
| `N`                        | `int`                                        |     `50`      | Number of memory cells (antibodies) in the population.                                                                                     |
| `n_clone`                  | `int`                                        |     `10`      | Number of clones generated for each selected memory cell.                                                                                  |
| `top_clonal_memory_size`   | `int`                                        |      `5`      | Number of highest-affinity antibodies selected per antigen for cloning and mutation.                                                       |
| `n_diversity_injection`    | `int`                                        |      `5`      | Number of new random memory cells injected to maintain diversity.                                                                          |
| `affinity_threshold`       | `float`                                      |     `0.5`     | Threshold for affinity (similarity) to determine cell suppression or selection.                                                            |
| `suppression_threshold`    | `float`                                      |     `0.5`     | Threshold for suppressing similar memory cells                                                                                             |
| `mst_inconsistency_factor` | `float`                                      |     `2.0`     | Factor used to determine which edges in the **Minimum Spanning Tree (MST)** are considered inconsistent.                                   |
| `max_iterations`           | `int`                                        |     `10`      | Maximum number of training iterations.                                                                                                     |
| `k`                        | `int`                                        |      `3`      | The number of K nearest neighbors that will be used to choose a label in the prediction.                                                   |
| `metric`                   | [`MetricType`](../utils/types.md#metrictype) | `"euclidean"` | Distance metric used to compute similarity between memory cells                                                                            |
| `seed`                     | `Optional[int]`                              |    `None`     | Seed for random generation.                                                                       |
| `use_mst_clustering`       | `bool`                                       |    `True`     | If ``True``, performs clustering with **Minimum Spanning Tree** (MST). If ``False``, does not perform clustering and predict returns None. |
| `p`                        | `float`                                      |     `2.0`     | This parameter stores the value of `p` used in the Minkowski distance.                                                                   |

## Attributes

| Name                    | Type                    | Default | Description                                                                        |
|-------------------------|-------------------------|:-------:|------------------------------------------------------------------------------------|
| `memory_network`        | `Dict[int, List[Cell]]` |    -    | The immune network representing clusters.                                          |
| `population_antibodies` | `Optional[npt.NDArray]` |    -    | The set of memory antibodies.                                                      |
| `mst`                   | `dict`                  |    -    | The Minimum Spanning Tree and its statistics (graph, mean_distance, std_distance). |

---

## Public Methods

### fit

```python
def fit(self, X: Union[npt.NDArray, list], verbose: bool = True) -> AiNet:
    ...
```

Train the AiNet model on input data.

**Parameters**

| Name      | Type                       | Default | Description                                                                  |
|-----------|----------------------------|:-------:|------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |    -    | Input data used for training the model.                                      |
| `verbose` | `bool`                     | `True`  | Feedback from the progress bar showing current training interaction details. |

**Raises**

* `TypeError` - If X is not a ndarray or list.
* [`UnsupportedTypeError`](../exceptions.md#unsupportedtypeerror) - If the data type of the vector is not supported.

**Returns**

`AiNet` - Returns the instance of the class that implements this method.

---

### predict

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
    ...
```

Predict cluster labels for input data.

**Parameters**

| Name | Type                       | Default | Description      |
|------|----------------------------|:-------:|------------------|
| `X`  | `Union[npt.NDArray, list]` |    -    | Data to predict. |

**Raises**

* `TypeError` - If X is not a ndarray or list.
* `ValueError` - If the array contains values other than 0 and 1.
* [`FeatureDimensionMismatch`](../exceptions.md#featuredimensionmismatch) - If the number of features in X does not match the expected number.
* [`ModelNotFittedError`](../exceptions.md#modelnotfittederror) - If the mode has not yet been adjusted and does not have defined memory cells, it is not able to predictions

**Returns**

**predictions** : `npt.NDArray` - Predicted cluster labels, or None if clustering is disabled.

---

### update_clusters

```python
def update_clusters(self, mst_inconsistency_factor: Optional[float] = None):
    ...
```
Partition the clusters based on the MST inconsistency factor.

Uses the precomputed Minimum Spanning Tree (MST) of the antibody population
to redefine clusters. Edges whose weights exceed the mean plus the
`mst_inconsistency_factor` multiplied by the standard deviation of MST edge
weights are removed. Each connected component after pruning is treated as a
distinct cluster.

**Parameters**

| Name                       | Type    | Default | Description                                              |
|----------------------------|---------|:-------:|----------------------------------------------------------|
| `mst_inconsistency_factor` | `float` | `None`  | If provided, overrides the current inconsistency factor. |

**Raises**

* `ValueError`
  * If the Minimum Spanning Tree (MST) has not yet been created
  * If Population of antibodies is empty
  * If MST statistics (mean or std) are not available.

**Updates**

* **memory_network** : `dict[int, npt.NDArray]` -
    Dictionary mapping cluster labels to antibody arrays.
* **labels** : `list` - List of cluster labels.

---

## Extended Example

Complete usage examples are available in the Jupyter Notebooks:

- [**Random Example**](../../../../examples/en/clustering/AiNet/example_with_randomly_generated_dataset.ipynb)
- [**Geyser Dataset Example**](../../../../examples/en/clustering/AiNet/geyser_dataBase_example.ipynb)

---

## References

[^1]: De Castro, Leandro & José, Fernando & von Zuben, Antonio Augusto. (2001). aiNet: An Artificial Immune Network for Data Analysis.
    Available at: https://www.researchgate.net/publication/228378350_aiNet_An_Artificial_Immune_Network_for_Data_Analysis

[^2]: SciPy Documentation. *Minimum Spanning Tree*.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree
