
# RNSA (Real-Valued Negative Selection Algorithm)

This class extends the [**Base**](../../advanced-guides/base/classifier.md) class.

## Constructor RNSA

The ``RNSA`` class has the purpose of classifying and identifying anomalies through the self and not self methods.

**Attributes:**

* **N** (``int``): Number of detectors. Defaults to ``100``.
* **r** (``float``): Radius of the detector. Defaults to ``0.05``.
* **k** (``int``): Number of neighbors near the randomly generated detectors to perform the distance average calculation. Defaults to ``1``.
* **metric** (``str``): Way to calculate the distance between the detector and the sample:

  * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression: √( (X₁ - X₂)² + (Y₁ - Y₂)² + ... + (Yn - Yn)²).
  * ``'minkowski'`` ➜ The calculation of the distance is given by the expression: ( |X₁ - Y₁|p + |X₂ - Y₂|p + ... |Xn - Yn|p) ¹/ₚ , In this project ``p == 2``.
  * ``'manhattan'`` ➜ The calculation of the distance is given by the expression: ( |X₁ - X₂| + |Y₁ - Y₂| + ...+ |Yn - Yn₂|) .

    Defaults to ``'euclidean'``.

* **max_discards** (``int``): This parameter indicates the maximum number of consecutive detector discards, aimed at preventing a possible infinite loop in case a radius is defined that cannot generate non-self detectors.
* **seed** (``int``): Seed for the random generation of values in the detectors. Defaults to ``None``.

* **algorithm** (``str``), Set the algorithm version:

  * ``'default-NSA'``: Default algorithm with fixed radius.
  * ``'V-detector'``: This algorithm is based on the article "[Real-Valued Negative Selection Algorithm with Variable-Sized Detectors](https://doi.org/10.1007/978-3-540-24854-5_30)", by Ji, Z., Dasgupta, D. (2004), and uses a variable radius for anomaly detection in feature spaces.

    Defaults to ``'default-NSA'``.

* **r_s** (``float``): rₛ Radius of the ``X`` own samples.
* ``**kwargs``:
  * *non_self_label* (``str``): This variable stores the label that will be assigned when the data has only one
    output class, and the sample is classified as not belonging to that class. Defaults to ``'non-self'``.
  * *cell_bounds* (``bool``): If set to ``True``, this option limits the generation of detectors to the space within
    the plane between 0 and 1. This means that any detector whose radius exceeds this limit is discarded,
    this variable is only used in the ``V-detector`` algorithm. Defaults to ``False``.

**Other variables initiated:**

* **detectors** (``dict``): This variable stores a list of detectors by class.
* **classes** (``npt.NDArray``): list of output classes.

---

### Method fit(...)

The ``fit(...)`` function generates the detectors for non-fits with respect to the samples:

```python
def fit(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list],
    verbose: bool = True,
) -> RNSA:
```

In it, training is performed according to ``X`` and ``y``, using the negative selection method(``NegativeSelect``).

The input parameters are:

* **X** (`Union[npt.NDArray, list]`): array with the characteristics of the samples with **N** samples (rows) and **N** characteristics (columns).
* **y** (`Union[npt.NDArray, list]`): array with the output classes arranged in **N** samples that are related to ``X``.
* **verbose** (`bool`): boolean with default value ``True``, determines if the feedback from the detector generation will be printed.

**Raises**

* ``TypeError``: If X or y are not ndarrays or have incompatible shapes.
* ``MaxDiscardsReachedError``: The maximum number of detector discards was reached during
  maturation. Check the defined radius value and consider reducing it.

**Returns**

 the instance of the class.

---

### Method predict(...)

The ``predict(...)`` function performs class prediction using the generated detectors:

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
```

**The input parameter is:**

* **X** (`Union[npt.NDArray, list]`): array with the characteristics for the prediction, with **N** samples (Rows) and **N** columns.

**Raises:**

* ``TypeError``: If X is not an ndarray or list.
* ``FeatureDimensionMismatch``: If the number of features in X does not match the expected number.
* ``ModelNotFittedError``: If the mode has not yet been adjusted and does not have defined detectors or classes, it is not able to predictions

**Returns:**

* C (``npt.NDArray``): A ndarray of the form ``C`` (n_samples), containing the predicted classes for ``X``.

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

### Method _checks_valid_detector(...)

The ``def _checks_valid_detector(...)`` function check if the detector has a valid non-proper r radius for the class.

```python
def _checks_valid_detector(
    self,
    x_class: npt.NDArray,
    vector_x: npt.NDArray
) -> Union[bool, tuple[bool, float]]:
```

**Parameters**

* **x_class** (`npt.NDArray`): Array ``x_class`` with the samples per class.
* **vector_x** (`npt.NDArray`): Randomly generated vector x candidate detector with values between[0, 1].

**Returns**
* **Validity** (``bool``): Returns whether the detector is valid or not.

---

### Method _compare_KnearestNeighbors_List(...)

The ``def _compare_KnearestNeighbors_List(...)`` function compares the distance of the k-nearest neighbors, so if the distance of the new sample is smaller, replaces ``k-1`` and sorts in ascending order:

```python
def _compare_knearest_neighbors_list(self, knn: list, distance: float) -> None:
```

**Parameters**
* **knn** (`list`): List of k-nearest neighbor distances.
* **distance** (`float`): Distance to check.

Returns a list of k-nearest neighbor distances.

---

### Method _compare_sample_to_detectors(...)

Function to compare a sample with the detectors, verifying if the sample is proper.
In this function, when there is class ambiguity, it returns the class that has the greatest average distance between the detectors.

```python
def _compare_sample_to_detectors(self, line: npt.NDArray) -> Optional[str]:
```

**Parameters**

* line (`npt.NDArray`): vector with N-features

**Returns**

The predicted class with the detectors or None if the sample does not qualify for any class.

---

### Method _detector_is_valid_to_Vdetector(...)

Check if the distance between the detector and the samples, minus the radius of the samples, is greater than the minimum radius.

```python
def _detector_is_valid_to_vdetector(
    self,
    distance: float,
    vector_x: npt.NDArray
) -> Union[bool, tuple[bool, float]]:
```

**Parameters**

* distance (``float``): minimum distance calculated between all samples.
* vector_x (``npt.NDArray``): randomly generated candidate detector vector x with values between 0 and 1.

**Returns:**

* ``False``: if the calculated radius is smaller than the minimum distance or exceeds the edge of the space, if this option is enabled.
* ``True`` and the distance minus the radius of the samples, if the radius is valid.`

---

### Method _distance(...)

The function ``def _distance(...)`` calculates the distance between two points using the technique defined in ``metric``, which are: ``'euclidean', 'norm_euclidean', or 'manhattan'``

```python
def _distance(self, u: npt.NDArray, v: npt.NDArray):
```

**Parameters**
* **u** (`npt.NDArray`): Coordinates of the first point.
* **v** (`npt.NDArray`): Coordinates of the second point.
**Returns:**
* distance (`float`): the distance between the two points.
