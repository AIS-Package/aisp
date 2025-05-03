# BNSA (Binary Negative Selection Algorithm)

This class extends the [**Base**](../../advanced-guides/Base%20Classes%20Reference/NegativeSelection.md) class.

## Constructor RNSA:

The ``BNSA`` (Binary Negative Selection Algorithm) class has the purpose of classifying and identifying anomalies through the self and not self methods.

**Attributes:**

* *N* (``int``): Number of detectors. Defaults to ``100``.
* *aff_thresh* (``float``): The variable represents the percentage of dissimilarity between the T cell and the own samples. The default value is 10% (0.1), while a value of 1.0 represents 100% dissimilarity.
* *max_discards* (``int``): This parameter indicates the maximum number of detector discards in sequence, which aims to avoid a
possible infinite loop if a radius is defined that it is not possible to generate non-self detectors. Defaults to ``100``.
* *seed* (``int``): Seed for the random generation of values in the detectors. Defaults to ``None``.
* no_label_sample_selection (``str``): Method for selecting labels for samples designated as non-members by all non-member detectors. **Available method types:**
    - (``max_average_difference``): Selects the class with the highest average difference among the detectors.
    - (``max_nearest_difference``): Selects the class with the highest difference between the nearest and farthest detector from the sample.

**Other variables initiated:**

* *detectors* (``dict``): This variable stores a list of detectors by class.

* *classes* (``npt.NDArray``): list of output classes.



### Function fit(...)

The ``fit(...)`` function generates the detectors for non-fits with respect to the samples:

```python
def fit(self, X: npt.NDArray, y: npt.NDArray):
```

In it, training is performed according to ``X`` and ``y``, using the negative selection method(``NegativeSelect``).

**The input parameters are:** 
* ``X``: array with the characteristics of the samples with **N** samples (rows) and **N** characteristics (columns). 

* ``y``: array with the output classes arranged in **N** samples that are related to ``X``.

* ``verbose``: boolean with default value ``True``, determines if the feedback from the detector generation will be printed.

**Raises**
* ``TypeError``: If X or y are not ndarrays or have incompatible shapes.
* ``MaxDiscardsReachedError``: The maximum number of detector discards was reached during
  maturation. Check the defined radius value and consider reducing it.


*Returns the instance of the class.*

---

### Function predict(...)

The ``predict(...)`` function performs class prediction using the generated detectors:

```python
def predict(self, X: npt.NDArray) -> npt.NDArray:
```

**The input parameter is:** 
* ``X``: array with the characteristics for the prediction, with **N** samples (Rows) and **N** columns.

**Raises:** 
* ``TypeError``: If X is not an ndarray or list.
* ``FeatureDimensionMismatch``: If the number of features in X does not match the expected number.
* ``ValueError``: X contains values that are not composed only of 0 and 1.

**Returns:** 
* ``C``: prediction array, with the output classes for the given characteristics.
* ``None``: if there are no detectors.

---

### Function score(...):

The function ``score(...)`` calculates the accuracy of the trained model by making predictions and computing accuracy.

```python
def score(self, X: npt.NDArray, y: list) -> float:
```

It returns the accuracy as a float type.

---

## Private Methods

---

### Function __assign_class_to_non_self_sample(...):

This function determines the class of a sample when all detectors classify it as "non-self". Classification is performed using the ``max_average_difference`` and ``max_nearest_difference`` methods.

```python
def __assign_class_to_non_self_sample(self, line, c) -> npt.NDArray:
```
**The input parameter is:** 

* ***line*** (``list``): Sample to be classified.
* ***c*** (``npt.NDArray``): List of predictions to be updated with the new classification.

**Returns:** 

``npt.NDArray``: The list of predictions `c` updated with the class assigned to the sample.

--

### Function __slice_index_list_by_class(...):

The function ``__slice_index_list_by_class(...)``, separates the indices of the lines according to the output class, to go through the sample array, only in the positions that the output is the class that is being trained:

```python
def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Returns a dictionary with the classes as key and the indices in ``X`` of the samples.
