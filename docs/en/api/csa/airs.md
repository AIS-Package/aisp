---
id: airs
sidebar_label: AIRS
keywords:
    - classification
    - artificial immune recognition system
    - memory cells
    - k-nn
    - supervised learning
tags:
    - classification
    - supervised
    - clonal selection
---

# AIRS

Artificial Immune Recognition System (AIRS)

:::tip[Inheritance]

This class extends [BaseClassifier](../base/base-classifier.md)

:::


> **Module:** `aisp.csa`  
> **Import:** `from aisp.csa import AIRS`

---

## Overview

The Artificial Immune Recognition System (AIRS) is a classification algorithm inspired by the
clonal selection process of the biological immune system. This implementation is based on the
simplified AIRS2 version described in [^1]. The algorithm has been adapted to support both
real-valued (continuous) and binary feature datasets.

:::note

This implementation is inspired by AIRS2, a simplified version of the original AIRS algorithm.
Introducing adaptations to handle continuous and binary datasets.

Based on Algorithm 16.5 from Brabazon et al. [^1]

Related and noteworthy works: access here [^2].

:::

---

## Example

```python
import numpy as np
from aisp.csa import AIRS

np.random.seed(1)
# Generating training data
a = np.random.uniform(high=0.5, size=(50, 2))
b = np.random.uniform(low=0.51, size=(50, 2))
x_train = np.vstack((a, b))
y_train = [0] * 50 + [1] * 50
# AIRS Instance
airs = AIRS(n_resources=5, rate_clonal=5, rate_hypermutation=0.65, seed=1)
airs = airs.fit(x_train, y_train, verbose=False)
x_test = [
    [0.15, 0.45],  # Expected: Class 0
    [0.85, 0.65],  # Esperado: Classe 1
]
y_pred = airs.predict(x_test)
print(y_pred)
```
Output:
```bash
[0 1]
```


---

## Constructor Parameters

| Name                        | Type    |    Default    | Description                                                                                                                                       |
|-----------------------------|---------|:-------------:|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `n_resources`               | `float` |     `10`      | Total amount of available resources.                                                                                                              |
| `rate_clonal`               | `float` |     `10`      | Maximum number of possible clones of a class. This quantity is multiplied by (cell_stimulus * rate_hypermutation) to define the number of clones. |
| `rate_mc_init`              | `float` |     `0.2`     | Percentage of samples used to initialize memory cells.                                                                                            |
| `rate_hypermutation`        | `float` |    `0.75`     | The rate of mutated clones derived from rate_clonal as a scalar factor.                                                                           |
| `affinity_threshold_scalar` | `float` |    `0.75`     | Normalized affinity threshold.                                                                                                                    |
| `k`                         | `int`   |      `3`      | The number of K nearest neighbors that will be used to choose a label in the prediction.                                                          |
| `max_iters`                 | `int`   |     `100`     | Maximum number of interactions in the refinement process of the ARB set exposed to aᵢ.                                                            |
| `resource_amplified`        | `float` |     `1.0`     | Resource consumption amplifier is multiplied with the incentive to subtract resources.                                                            |
| `metric`                    | `str`   | `"euclidean"` | Distance metric used to compute affinity between cells and samples.                                                                               |
| `seed`                      | `int`   |    `None`     | Seed for the random generation of detector values. Defaults to None.                                                                              |
| `p`                         | `float` |      `2`      | This parameter stores the value of ``p`` used in the Minkowski distance.                                                                          |

## Attributes

| Name           | Type                                      | Default | Description                                             |
|----------------|-------------------------------------------|:-------:|---------------------------------------------------------|
| `cells_memory` | `Optional[Dict[str \| int, list[BCell]]]` |    -    | Dictionary of trained memory cells, organized by class. |


---

## Public Methods

### fit

```python
def fit(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list],
    verbose: bool = True,
) -> AIRS:
    ...
```
Fit the model to the training data using the AIRS.

The function ``fit(...)``, performs the training according to ``X`` and ``y``, using the
method AIRS.

**Parameters**

| Name      | Type                       | Default | Description                                                                                       |
|-----------|----------------------------|:-------:|---------------------------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |    -    | Training array, containing the samples and their characteristics, Shape: (n_samples, n_features). |
| `y`       | `Union[npt.NDArray, list]` |    -    | Array of target classes of ``X`` with (``n_samples``).                                            |
| `verbose` | `bool`                     | `True`  | Feedback on which sample aᵢ the memory cells are being generated.                                 |

**Returns**

`AIRS` - Returns the instance itself.

---

### predict

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
    ...
```
Predict class labels based on the memory cells created during training.

This method uses the trained memory cells to perform classification of the input data
using the k-nearest neighbors approach.

**Parameters**

| Name | Type                       | Default | Description                                                       |
|------|----------------------------|:-------:|-------------------------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |    -    | Array with input samples with  Shape: (``n_samples, n_features``) |

**Raises**

* `TypeError` - If X is not a ndarray or list.
* [`FeatureDimensionMismatch`](../exceptions.md#featuredimensionmismatch) - If the number of features in X does not match the expected number.
* [`ModelNotFittedError`](../exceptions.md#modelnotfittederror) - If the mode has not yet been adjusted and does not have defined memory cells, it is not able to predictions

**Returns**

**C** : `npt.NDArray` - An ndarray of the form ``C`` (``n_samples``), containing the predicted classes for ``X``.

---

## Extended Example

Complete usage examples are available in the Jupyter Notebooks:

- [**Iris Dataset Example**](../../../../examples/en/classification/AIRS/iris_dataBase_example_en.ipynb)
- [**Geyser Dataset Example**](../../../../examples/en/classification/AIRS/geyser_dataBase_example_en.ipynb)
- [**Mushrooms Dataset Example**](../../../../examples/en/classification/AIRS/mushrooms_dataBase_example_en.ipynb)
- [**Random Dataset Example**](../../../../examples/en/classification/AIRS/example_with_randomly_generated_dataset-en.ipynb)


---

## References

[^1]: Brabazon, A., O'Neill, M., & McGarraghy, S. (2015). Natural Computing Algorithms. In Natural Computing Series.
    Springer Berlin Heidelberg. [https://doi.org/10.1007/978-3-662-43631-8](https://doi.org/10.1007/978-3-662-43631-8)

[^2]: AZZOUG, Aghiles. Artificial Immune Recognition System V2. Available at:
    [https://github.com/AghilesAzzoug/Artificial-Immune-System](https://github.com/AghilesAzzoug/Artificial-Immune-System)
