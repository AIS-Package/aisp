# Metrics

## def accuracy_score(...)

```python
def accuracy_score(
        y_true: Union[npt.NDArray, list],
        y_pred: Union[npt.NDArray, list]
) -> float
```

Function to calculate precision accuracy based on lists of true labels and
predicted labels.

**Parameters**:
* **_y_true_** (``Union[npt.NDArray, list]``): Ground truth (correct) labels.
    Expected to be of the same length as `y_pred`.
* **_y_pred_** (``Union[npt.NDArray, list]``): Predicted labels. Expected to
    be of the same length as `y_true`.

Returns:
* **_Accuracy_** (``float``): The ratio of correct predictions to the total
number of predictions.

**Raises**:
* `ValueError`: If `y_true` or `y_pred` are empty or if they do not have the same length.

---

# Multiclass

## def slice_index_list_by_class(...)

```python
def slice_index_list_by_class(classes, y: npt.NDArray) -> dict
```

The function ``__slice_index_list_by_class(...)``, separates the indices of the lines
according to the output class, to loop through the sample array, only in positions where
the output is the class being trained.

**Parameters:**
* **_y_** (npt.NDArray): Receives a ``y``[``N sample``] array with the output classes of the
    ``X`` sample array.

**returns:**
* `dict`: A dictionary with the list of array positions(``y``), with the classes as key.

---

# Sanitizers


## def sanitize_choice(...)

```python
def sanitize_choice(value: T, valid_choices: Iterable[T], default: T) -> T
```

The function ``sanitize_choice(...)``, returns the value if it is present in the set of valid choices; otherwise, returns the default value.


**Parameters:**
* ***value*** (``T``): The value to be checked.
* ***valid_choices*** (``Iterable[T]``): A collection of valid choices.
* ***default***:  The default value to be returned if ``value`` is not in ``valid_choices``.


**Returns:**
* `T`: The original value if valid, or the default value if not.

---

## def sanitize_param(...)

```python
def sanitize_param(value: T, default: T, condition: Callable[[T], bool]) -> T:
```

The function ``sanitize_param(...)``, returns the value if it satisfies the specified condition; otherwise, returns the default value.

**Parameters:**
* value (``T``): The value to be checked.
* default (``T``): The default value to be returned if the condition is not satisfied.
* condition (``Callable[[T], bool]``): A function that takes a value and returns a boolean, determining if the value is valid.


**Returns:**
* `T`: The original value if the condition is satisfied, or the default value if not.

---

## def sanitize_seed(...)

```python
def sanitize_seed(seed: Any) -> Optional[int]:
```

The function ``sanitize_param(...)``, returns the seed if it is a non-negative integer; otherwise, returns None.

**Parameters:**
* seed (``Any``): The seed value to be validated.

**Returns:**
* ``Optional[int]``: The original seed if it is a non-negative integer, or ``None`` if it is invalid.

# Distance

Utility functions for normalized distance between arrays with numba decorators.

## def hamming(...)

```python
def hamming(u: npt.NDArray, v: npt.NDArray) -> np.float64:
```

The function to calculate the normalized Hamming distance between two points.
    
$((x₁ ≠ x₂) + (y₁ ≠ y₂) + ... + (yn ≠ yn)) / n$


**Parameters:**
* u (``npt.NDArray``): Coordinates of the first point.
* v (``npt.NDArray``): Coordinates of the second point.

**Returns:**
* Distance (``float``) between the two points.

---

## def euclidean(...)

```python
def euclidean(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> np.float64:
```

Function to calculate the normalized Euclidean distance between two points.

$√( (x₁ - x₂)² + (y₁ - y₂)² + ... + (yn - yn)²)$



**Parameters:**
* u (``npt.NDArray``): Coordinates of the first point.
* v (``npt.NDArray``): Coordinates of the second point.

**Returns:**
* Distance (``float``) between the two points.

---

## def cityblock(...)

```python
def cityblock(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> np.float64:
```

Function to calculate the normalized Manhattan distance between two points.
    
$(|x₁ - x₂| + |y₁ - y₂| + ... + |yn - yn|) / n$


**Parameters:**
* u (``npt.NDArray``): Coordinates of the first point.
* v (``npt.NDArray``): Coordinates of the second point.

**Returns:**
* Distance (``float``) between the two points.

---

## def minkowski(...)

```python
def minkowski(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64], p: float = 2.0):
```

Function to calculate the normalized Minkowski distance between two points.
    
$(( |X₁ - Y₁|p + |X₂ - Y₂|p + ... + |Xn - Yn|p) ¹/ₚ) / n$


**Parameters:**
* u (``npt.NDArray``): Coordinates of the first point.
* v (``npt.NDArray``): Coordinates of the second point.
* p float: The p parameter defines the type of distance to be calculated:
    - p = 1: **Manhattan** distance — sum of absolute differences.
    - p = 2: **Euclidean** distance — sum of squared differences (square root).
    - p > 2: **Minkowski** distance with an increasing penalty as p increases.

**Returns:**
* Distance (``float``) between the two points.

---

## def compute_metric_distance(...)

```python
def compute_metric_distance(
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    metric: int,
    p: np.float64 = 2.0
) -> np.float64:
```

Function to calculate the distance between two points by the chosen ``metric``.

**Parameters:**
* u (``npt.NDArray``): Coordinates of the first point.
* v (``npt.NDArray``): Coordinates of the second point.
* metric (``int``): Distance metric to be used. Available options: [0 (Euclidean), 1 (Manhattan), 2 (Minkowski)]
* p (``float``): Parameter for the Minkowski distance (used only if `metric` is "minkowski").

**Returns:**
* Distance (``double``) between the two points with the selected metric.

---

## def min_distance_to_class_vectors(...)

```python
def min_distance_to_class_vectors(
    x_class: npt.NDArray,
    vector_x: npt.NDArray,
    metric: int,
    p: float = 2.0
) -> float:
```

Calculates the minimum distance between an input vector and the vectors of a class.


**Parameters:**
* x_class (``npt.NDArray``): Array containing the class vectors to be compared with the input vector. Expected shape: (n_samples, n_features).
* vector_x (``npt.NDArray``): Vector to be compared with the class vectors. Expected shape: (n_features,).
* metric (``int``): Distance metric to be used. Available options: [0 (Euclidean), 1 (Manhattan), 2 (Minkowski)]
* p (``float``): Parameter for the Minkowski distance (used only if `metric` is "minkowski").

**Returns:**
* float: The minimum distance calculated between the input vector and the class vectors.
* Returns -1.0 if the input dimensions are incompatible.

---

## def get_metric_code(...)

```python
def get_metric_code(metric: str) -> int:
```
Returns the numeric code associated with a distance metric.

**Parameters:**
* metric (str): Name of the metric. Can be "euclidean", "manhattan", "minkowski" or "hamming".

**Raises**
----------
* ``ValueError``: If the metric provided is not supported

**Returns:**
* ``int``: Numeric code corresponding to the metric.

---

# Validation

## def detect_vector_data_type(...)

```python
def detect_vector_data_type(
    vector: npt.NDArray
) -> FeatureType:
```

Detects the type of data in a given vector.

This function analyzes the input vector and classifies its data as one of the supported types:

* **binary**: Boolean values (`True`/`False`) or integer `0`/`1`.
* **continuous**: Float values within the normalized range `[0.0, 1.0]`.
* **ranged**: Float values outside the normalized range.

### Parameters

* `vector` (`npt.NDArray`): An array containing the data to be classified.

### Returns

* `FeatureType` (`Literal["binary-features", "continuous-features", "ranged-features"]`): The detected type of data in the vector.

### Raises

* `UnsupportedDataTypeError`: Raised if the vector contains an unsupported data type.
