# BNSA (Binary Negative Selection Algorithm)

This class extends the [**Base**](../../advanced-guides/base/classifier.md) class.

## Constructor BNSA

The ``BNSA`` (Binary Negative Selection Algorithm) class has the purpose of classifying and identifying anomalies through the self and not self methods.

**Attributes:**

* *N* (``int``): Number of detectors. Defaults to ``100``.
* *aff_thresh* (``float``): The variable represents the percentage of similarity between the T cell and the own samples. The default value is 10% (0.1), while a value of 1.0 represents 100% dissimilarity.
* *max_discards* (``int``): This parameter indicates the maximum number of detector discards in sequence, which aims to avoid a
possible infinite loop if a radius is defined that it is not possible to generate non-self detectors. Defaults to ``1000``.
* *seed* (``int``): Seed for the random generation of values in the detectors. Defaults to ``None``.
* no_label_sample_selection (``str``): Method for selecting labels for samples designated as non-members by all non-member detectors. **Available method types:**
  * (``max_average_difference``): Selects the class with the highest average difference among the detectors.
  * (``max_nearest_difference``): Selects the class with the highest difference between the nearest and farthest detector from the sample.

**Other variables initiated:**

* *detectors* (``dict``): This variable stores a list of detectors by class.
* *classes* (``npt.NDArray``): list of output classes.

### Method fit(...)

The ``fit(...)`` function generates the detectors for non-fits with respect to the samples:

```python
def fit(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list],
    verbose: bool = True,
) -> BNSA:
```

In it, training is performed according to ``X`` and ``y``, using the negative selection method(``NegativeSelect``).

**Parameters**

* **X** (`Union[npt.NDArray, list]`): array with the characteristics of the samples with **N** samples (rows) and **N** characteristics (columns).

* **y** (`Union[npt.NDArray, list]`): array with the output classes arranged in **N** samples that are related to ``X``.

* **verbose** (`bool`): boolean with default value ``True``, determines if the feedback from the detector generation will be printed.

**Raises**

* ``TypeError``: If X or y are not ndarrays or have incompatible shapes.
* ``ValueError``: If the array contains values other than 0 and 1.
* ``MaxDiscardsReachedError``: The maximum number of detector discards was reached during
  maturation. Check the defined radius value and consider reducing it.

*Returns the instance of the class.*

---

### Method predict(...)

The ``predict(...)`` function performs class prediction using the generated detectors:

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
```

**Parameters**

* **X** (`Union[npt.NDArray, list]`): array with the characteristics for the prediction, with **N** samples (Rows) and **N** columns.

**Raises:**

* ``TypeError``: If X is not an ndarray or list.
* ``ValueError``: X contains values that are not composed only of 0 and 1.
* ``FeatureDimensionMismatch``: If the number of features in X does not match the expected number.
* ``ModelNotFittedError``: If the mode has not yet been adjusted and does not have defined detectors or classes, it is not able to predictions

**Returns:**

* **C** (`npt.NDArray`): A ndarray of the form ``C`` (``n_samples``), containing the predicted classes for ``X``.

---

### Method score(...)

The function ``score(...)`` calculates the accuracy of the trained model by making predictions and computing accuracy.

```python
def score(self, X: npt.NDArray, y: list) -> float:
```

It returns the accuracy as a float type.

---

## Private Methods

---

### Method _assign_class_to_non_self_sample(...)

This function determines the class of a sample when all detectors classify it as "non-self". Classification is performed using the ``max_average_difference`` and ``max_nearest_difference`` methods.

```python
def _assign_class_to_non_self_sample(self, line: npt.NDArray, c: list):
```

**Parameters**

* **line** (``npt.NDArray``): Sample to be classified.
* **c** (``list``): List of predictions to be updated with the new classification.

**Raises**
* `ValueError`: If detectors is not initialized.

**Returns:**

``npt.NDArray``: The list of predictions `c` updated with the class assigned to the sample.
