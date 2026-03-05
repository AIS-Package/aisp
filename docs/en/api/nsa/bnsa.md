---
id: bnsa
sidebar_label: BNSA
keywords:
    - negative selection
    - binary features
    - anomaly detection
    - non-self recognition
    - pattern recognition
    - classification
    - multiclass
tags:
    - classification
    - supervised
    - negative selection
    - binary-features
    - anomaly detection
---

# BNSA

Binary Negative Selection Algorithm (BNSA).

:::tip[Inheritance]
This class extends [BaseClassifier](../base/base-classifier.md)
:::


> **Module:** `aisp.nsa`  
> **Import:** `from aisp.nsa import BNSA`

---

## Overview

Algorithm for classification and anomaly detection Based on self or not self
discrimination, inspired by Negative Selection Algorithm.

:::note

The **Binary Negative Selection Algorithm (BNSA)** is based on the original proposal by
Forrest et al. (1994) [^1], originally developed for computer security. In the adaptation, the
algorithm use bits arrays, and it has support for multiclass classification.

:::

:::warning

High `aff_thresh` values may prevent the generation of valid non-self detectors

:::

---

## Example

```python
from aisp.nsa import BNSA
# Binary 'self' samples
x_train  = [
    [0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0]
]
y_train = ['self', 'self', 'self', 'self', 'self', 'self']
bnsa = BNSA(aff_thresh=0.55, seed=1)
bnsa = bnsa.fit(x_train, y_train, verbose=False)
# samples for testing
x_test = [
...     [1, 1, 1, 1, 1], # Sample of Anomaly
...     [0, 1, 0, 1, 0], # self sample
... ]
y_prev = bnsa.predict(X=x_test)
print(y_prev)
```
**Output**
```bash
['non-self' 'self']
```

---

## Constructor Parameters

| Name                        | Type            |          Default           | Description                                                                                                                                                                                               |
|-----------------------------|-----------------|:--------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `N`                         | `int`           |           `100`            | Number of detectors.                                                                                                                                                                                      |
| `aff_thresh`                | `float`         |           `0.1`            | The variable represents the percentage of similarity between the T cell and the own samples. The default value is 10% (0.1), while a value of 1.0 represents 100% similarity.                             |
| `max_discards`              | `int`           |           `1000`           | This parameter indicates the maximum number of detector discards in sequence, which aims to avoid a possible infinite loop if a radius is defined that it is not possible to generate non-self detectors. |
| `seed`                      | `Optional[int]` |           `None`           | Seed for the random generation of values in the detectors.                                                                                                                                                |
| `no_label_sample_selection` | `str`           | `'max_average_difference'` | Method for selecting labels for samples designated as non-self by all detectors.                                                                                                                          |

## Attributes

| Name        | Type                                                | Default | Description                                |
|-------------|-----------------------------------------------------|:-------:|--------------------------------------------|
| `detectors` | `Optional[Dict[str \| int, npt.NDArray[np.bool_]]]` |    -    | The trained detectors, organized by class. |

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

Training according to X and y, using the method negative selection method.

**Parameters**

| Name      | Type                       | Default | Description                                                                                       |
|-----------|----------------------------|:-------:|---------------------------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |    -    | Training array, containing the samples and their characteristics, Shape: (n_samples, n_features). |
| `y`       | `Union[npt.NDArray, list]` |    -    | Array of target classes of ``X`` with (``n_samples``).                                            |
| `verbose` | `bool`                     | `True`  | Feedback from detector generation to the user.                                                    |


**Raises**

* `TypeError` - If X or y are not ndarrays or have incompatible shapes.
* `ValueError` - If the array contains values other than 0 and 1.
* `MaxDiscardsReachedError` - The maximum number of detector discards was reached during maturation. Check the defined radius value and consider reducing it.

**Returns**

`BNSA` - Returns the instance itself.

---

### predict

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
    ...
```

Prediction of classes based on detectors created after training.

**Parameters**

| Name | Type                       | Default | Description                                                       |
|------|----------------------------|:-------:|-------------------------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |    -    | Array with input samples with  Shape: (``n_samples, n_features``) |

**Raises**

* `TypeError` - If X is not a ndarray or list.
* `ValueError` - If the array contains values other than 0 and 1.
* `FeatureDimensionMismatch` - If the number of features in X does not match the expected number.
* `ModelNotFittedError` - If the mode has not yet been adjusted and does not have defined detectors or classes, it is not able to predictions

**Returns**

**C** : `npt.NDArray` - An ndarray of the form ``C`` (``n_samples``), containing the predicted classes for ``X``.

---

## Extended Example

Complete usage examples are available in the Jupyter Notebooks:

- [**Mushrooms Dataset Example**](../../../../examples/en/classification/BNSA/mushrooms_dataBase_example_en.ipynb)
- [**Random Dataset Example**](../../../../examples/en/classification/BNSA/example_with_randomly_generated_dataset-en.ipynb)

---

## References

[^1]: S. Forrest, A. S. Perelson, L. Allen and R. Cherukuri, "Self-nonself discrimination in
    a computer," Proceedings of 1994 IEEE Computer Society Symposium on Research in Security
    and Privacy, Oakland, CA, USA, 1994, pp. 202-212,
    doi: https://dx.doi.org/10.1109/RISP.1994.296580.
