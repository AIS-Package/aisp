---
id: rnsa
sidebar_label: RNSA
keywords:
    - negative selection
    - anomaly detection
    - non-self recognition
    - pattern recognition
    - classification
    - real-valued
    - v-detector
    - multiclass
tags:
    - classification
    - supervised
    - negative selection
    - real-valued
    - anomaly detection
---

# RNSA

Real-Valued Negative Selection Algorithm (RNSA).

:::tip[Inheritance]

This class extends [BaseClassifier](../base/base-classifier.md)

:::


> **Module:** `aisp.nsa`  
> **Import:** `from aisp.nsa import RNSA`

---

## Overview

Algorithm for classification and anomaly detection Based on self or not self
discrimination, inspired by Negative Selection Algorithm.

:::note

This algorithm has two different versions: one based on the canonical version [^1] and another
with variable radius detectors [^2]. Both are adapted to work with multiple classes and have
methods for predicting data present in the non-self region of all detectors and classes.

:::

:::warning

The parameters `r` and `r_s` can prevent the generation of valid detectors. A very small `r`
value can limit coverage, while a very high one can hinder the generation of valid detectors.
Similarly, a high r_s can restrict detector creation. Thus, proper adjustment of `r` and `r_s`
is essential to ensure good model performance.

:::

---

## Example

```python
import numpy as np
from aisp.nsa import RNSA

np.random.seed(1)
class_a = np.random.uniform(high=0.5, size=(50, 2))
class_b = np.random.uniform(low=0.51, size=(50, 2))
```

**Example 1:** Multiclass classification (RNSA supports two or more classes)

```python
x_train = np.vstack((class_a, class_b))
y_train = ['a'] * 50 + ['b'] * 50
rnsa = RNSA(N=150, r=0.3, seed=1)
rnsa = rnsa.fit(x_train, y_train, verbose=False)
x_test = [
    [0.15, 0.45],  # Expected: Class 'a'
    [0.85, 0.65],  # Expected: Class 'b'
]
y_pred = rnsa.predict(x_test)
print(y_pred)
```

**Output**

```bash
['a' 'b']
```

**Example 2:** Anomaly Detection (self/non-self)

```python
rnsa = RNSA(N=150, r=0.3, seed=1)
rnsa = rnsa.fit(X=class_a, y=np.array(['self'] * 50), verbose=False)
y_pred = rnsa.predict(class_b[:5])
print(y_pred)
```

**Output**

```bash
['non-self' 'non-self' 'non-self' 'non-self' 'non-self']
```

---

## Constructor Parameters

| Name             | Type                                      |     Default     | Description                                                                                                                                                                                                                                              |
|------------------|-------------------------------------------|:---------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `N`              | `int`                                     |      `100`      | Number of detectors.                                                                                                                                                                                                                                     |
| `r`              | `float`                                   |     `0.05`      | Radius of the detector.                                                                                                                                                                                                                                  |
| `r_s`            | `float`                                   |    `0.0001`     | rₛ Radius of the `X` own samples.                                                                                                                                                                                                                        |
| `k`              | `int`                                     |       `1`       | Number of neighbors near the randomly generated detectors to perform the distance average calculation.                                                                                                                                                   |
| `metric`         | `{"euclidean", "minkowski", "manhattan"}` |  `'euclidean'`  | Distance metric used to compute the distance between the detector and the sample.                                                                                                                                                                        |
| `max_discards`   | `int`                                     |     `1000`      | This parameter indicates the maximum number of consecutive detector discards, aimed at preventing a possible infinite loop in case a radius is defined that cannot generate non-self detectors.                                                          |
| `seed`           | `Optional[int]`                           |     `None`      | Seed for the random generation of values in the detectors.                                                                                                                                                                                               |
| `algorithm`      | `{"default-NSA", "V-detector"}`           | `'default-NSA'` | Set the algorithm version                                                                                                                                                                                                                                |
| `non_self_label` | `str`                                     |  `'non-self'`   | This variable stores the label that will be assigned when the data has only one output class, and the sample is classified as not belonging to that class.                                                                                               |
| `cell_bounds`    | `bool`                                    |     `False`     | If set to ``True``, this option limits the generation of detectors to the space within the plane between 0 and 1. This means that any detector whose radius exceeds this limit is discarded, this variable is only used in the ``V-detector`` algorithm. |
| `p`              | `bool`                                    |      `2.0`      | This parameter stores the value of `p` used in the Minkowski distance.                                                                                                                                                                                   |

## Attributes

| Name        | Type                                         | Default | Description                                |
|-------------|----------------------------------------------|:-------:|--------------------------------------------|
| `detectors` | `Optional[Dict[str \| int, list[Detector]]]` |    -    | The trained detectors, organized by class. |

---

## Public Methods

### fit

```python
def fit(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list],
    verbose: bool = True,
) -> BNSA:
    ...
```

Perform training according to X and y, using the negative selection method (NegativeSelect).

**Parameters**

| Name      | Type                       | Default | Description                                                                          |
|-----------|----------------------------|:-------:|--------------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |    -    | Training input samples. Each row corresponds to a samples and column to feature.     |
| `y`       | `Union[npt.NDArray, list]` |    -    | Target vector of shape (n_samples,). Must contain the same number of samples as `X`. |
| `verbose` | `bool`                     | `True`  | If True, prints training progress information.                                       |

**Returns**

| Type   | Description                  |
|--------|------------------------------|
| `Self` | Returns the instance itself. |


**Raises**

| Exception                                                             | Description                                                                                                                     |
|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| `TypeError`                                                           | If X or y are not ndarrays or have incompatible shapes.                                                                         |
| `ValueError`                                                          | If the array X fall outside the interval (0.0, 1.0).                                                                            |
| [`MaxDiscardsReachedError`](../exceptions.md#maxdiscardsreachederror) | The maximum number of detector discards was reached during maturation. Check the defined radius value and consider reducing it. |

---

### predict

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
    ...
```

Prediction of classes based on detectors created after training.

**Parameters**

| Name | Type                       | Default | Description                                                                |
|------|----------------------------|:-------:|----------------------------------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |    -    | Input samples. Must have the same number of features used during training. |

**Returns**

| Type          | Description                                                                         |
|---------------|-------------------------------------------------------------------------------------|
| `npt.NDArray` | An ndarray of the form `C` (`n_samples`), containing the predicted classes for `X`. |

**Raises**

| Exception                                                               | Description                                                                                                         |
|-------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| `TypeError`                                                             | If X is not a ndarray or list.                                                                                      |
| `ValueError`                                                            | If the array X fall outside the interval (0.0, 1.0).                                                                |
| [`FeatureDimensionMismatch`](../exceptions.md#featuredimensionmismatch) | If the number of features in X does not match the expected number.                                                  |
| [`ModelNotFittedError`](../exceptions.md#modelnotfittederror)           | If the mode has not yet been adjusted and does not have defined detectors or classes, it is not able to predictions |

---

## Extended Example

Complete usage examples are available in the Jupyter Notebooks:

- [**Iris Dataset Example**](../../../../examples/en/classification/RNSA/iris_dataBase_example_en.ipynb)
- [**Geyser Dataset Example**](../../../../examples/en/classification/RNSA/geyser_dataBase_example_en.ipynb)
- [**Random Dataset Example**](../../../../examples/en/classification/RNSA/example_with_randomly_generated_dataset-en.ipynb)

---

## References

[^1]: BRABAZON, Anthony; O'NEILL, Michael; MCGARRAGHY, Seán. Natural Computing
    Algorithms. [S. l.]: Springer Berlin Heidelberg, 2015. DOI 10.1007/978-3-662-43631-8.
    Disponível em: https://dx.doi.org/10.1007/978-3-662-43631-8.

[^2] Ji, Z.; Dasgupta, D. (2004). Real-Valued Negative Selection Algorithm with Variable-Sized Detectors.
    In *Lecture Notes in Computer Science*, vol. 3025. https://doi.org/10.1007/978-3-540-24854-5_30
