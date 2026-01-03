# Utils

Utility functions and helpers for development.

## Metrics

### def accuracy_score(...)

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

## Multiclass

## def predict_knn_affinity(...)

```python
def predict_knn_affinity(
    X: npt.NDArray,
    k: int,
    all_cell_vectors: List[Tuple[Union[str, int], npt.NDArray]],
    affinity_func: Callable[[npt.NDArray, npt.NDArray], float]
) -> npt.NDArray
```

Function to predict classes using k-nearest neighbors and trained cells.

**Parameters:**

* **_X_** (`npt.NDArray`): Input data to be classified.
* **_k_** (`int`): Number of nearest neighbors to consider for prediction.
* **_all_cell_vectors_** (`List[Tuple[Union[str, int], npt.NDArray]]`): List of tuples containing (class_name, cell vector) pairs.
* **_affinity_func_** (`Callable[[npt.NDArray, npt.NDArray], float]`): Function that takes two vectors and returns an affinity value.

**Returns:**

* `npt.NDArray`: Array of predicted labels for each sample in X, based on the k nearest neighbors.

---

## def slice_index_list_by_class(...)

```python
def slice_index_list_by_class(classes: Optional[Union[npt.NDArray, list]], y: npt.NDArray) -> dict
```

The function ``slice_index_list_by_class(...)``, separates the indices of the samples
according to their output class, to iterate through the sample array only at positions where
the output matches the class being trained.

**Parameters:**

* **_classes_** (`Optional[Union[npt.NDArray, list]]`): List with unique classes. If None, returns an empty dictionary.
* **_y_** (`npt.NDArray`): Array with output classes of the ``X`` sample array.

**Returns:**

* `dict`: A dictionary with the list of array positions keyed by class.

**Examples:**

```python
>>> import numpy as np
>>> labels = ['a', 'b', 'c']
>>> y = np.array(['a', 'c', 'b', 'a', 'c', 'b'])
>>> slice_index_list_by_class(labels, y)
{'a': [0, 3], 'b': [2, 5], 'c': [1, 4]}
```

---

The function ``__slice_index_list_by_class(...)``, separates the indices of the lines
according to the output class, to loop through the sample array, only in positions where
the output is the class being trained.

**Parameters:**

* **_y_** (npt.NDArray): Receives a ``y`` (``n_samples``) array with the output classes of the
    ``X`` sample array.

**returns:**

* `dict`: A dictionary with the list of array positions(``y``), with the classes as key.

---

## Sanitizers

### def sanitize_choice(...)

```python
def sanitize_choice(value: T, valid_choices: Iterable[T], default: T) -> T
```

The function ``sanitize_choice(...)``, returns the value if it is present in the set of valid choices; otherwise, returns the default value.

**Parameters:**

* _**value**_ (``T``): The value to be checked.
* _**valid_choices**_ (``Iterable[T]``): A collection of valid choices.
* _**default**_:  The default value to be returned if ``value`` is not in ``valid_choices``.

**Returns:**

* `T`: The original value if valid, or the default value if not.

---

### def sanitize_param(...)

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

### def sanitize_seed(...)

```python
def sanitize_seed(seed: Any) -> Optional[int]:
```

The function ``sanitize_param(...)``, returns the seed if it is a non-negative integer; otherwise, returns None.

**Parameters:**

* seed (``Any``): The seed value to be validated.

**Returns:**

* ``Optional[int]``: The original seed if it is a non-negative integer, or ``None`` if it is invalid.

### def sanitize_bounds(...)

```python
def sanitize_bounds(bounds: Any, problem_size: int) -> Dict[str, npt.NDArray[np.float64]]
```

The function ``sanitize_bounds(...)``, validate and normalize feature bounds.

**Parameters:**

* _**bounds**_ (``Any``): he input bounds, which must be either None or a dictionary with 'low' and 'high' keys.
* _**problem_size**_ (``int``): The expected length for the normalized bounds lists, corresponding to the number of features in the problem.

**Returns:**

* `Dict[str, list]`: Dictionary {'low': [low_1, ..., low_N], 'high': [high_1, ..., high_N]}.

## Distance

Utility functions for normalized distance between arrays with numba decorators.

### def hamming(...)

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

### def euclidean(...)

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

### def cityblock(...)

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

### def minkowski(...)

```python
def minkowski(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64], p: float = 2.0):
```

Function to calculate the normalized Minkowski distance between two points.

$(( |X₁ - Y₁|p + |X₂ - Y₂|p + ... + |Xn - Yn|p) ¹/ₚ) / n$

**Parameters:**

* u (``npt.NDArray``): Coordinates of the first point.
* v (``npt.NDArray``): Coordinates of the second point.
* p float: The p parameter defines the type of distance to be calculated:
  * p = 1: **Manhattan** distance — sum of absolute differences.
  * p = 2: **Euclidean** distance — sum of squared differences (square root).
  * p > 2: **Minkowski** distance with an increasing penalty as p increases.

**Returns:**

* Distance (``float``) between the two points.

---

### def compute_metric_distance(...)

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

### def min_distance_to_class_vectors(...)

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

### def get_metric_code(...)

```python
def get_metric_code(metric: str) -> int:
```

Returns the numeric code associated with a distance metric.

**Parameters:**

* metric (str): Name of the metric. Can be "euclidean", "manhattan", "minkowski" or "hamming".

**Raises**

* ``ValueError``: If the metric provided is not supported

**Returns:**

* ``int``: Numeric code corresponding to the metric.

---

## Validation

### def detect_vector_data_type(...)

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

#### Parameters

* `vector` (`npt.NDArray`): An array containing the data to be classified.

#### Returns

* `FeatureType` (`Literal["binary-features", "continuous-features", "ranged-features"]`): The detected type of data in the vector.

#### Raises

* `UnsupportedDataTypeError`: Raised if the vector contains an unsupported data type.

---

### def check_array_type(...)

```python
def check_array_type(x, name: str = "X") -> npt.NDArray:
```

Ensure X is a numpy array. Convert from list if needed.

#### Parameters

* `x` (`Any`): Array, containing the samples and their characteristics. Shape: (``n_samples, n_features``).
* `name` (`str`, default='X'): Variable name used in error messages.

#### Returns

* `npt.NDArray`: The converted or validated array.

#### Raises

* `TypeError`: If X or y are not ndarrays or have incompatible shapes.

---

### def check_shape_match(...)

```python
def check_shape_match(x: npt.NDArray, y: npt.NDArray):
```

Ensure X and y have compatible first dimensions.

#### Parameters

* `x` (`npt.NDArray`): Array, containing the samples and their characteristics. Shape: (``n_samples, n_features``).
* `y` (`npt.NDArray`): Array of target classes of `x` with (``n_samples``).

#### Raises

* `TypeError`: If x or y are not ndarrays or have incompatible shapes.

---

### def check_feature_dimension(...)

```python
def check_feature_dimension(x: npt.NDArray, expected: int):
```

Ensure X has the expected number of features.

#### Parameters

* `x` (`npt.NDArray`): Input array for prediction, containing the samples and their characteristics. Shape: (``n_samples, n_features``).
* `expected` (`int`): Expected number of features per sample (columns in X).

#### Raises

* `FeatureDimensionMismatch`: If the number of features in X does not match the expected number.

---

### def check_binary_array(...)

```python
def check_binary_array(x: npt.NDArray):
```

Ensure X contains only 0 and 1.

#### Raises

* `ValueError`: If feature_type is binary-features and X contains values that are not composed only of 0 and 1.

---

## Display

Utility Functions for Displaying Algorithm Information

### def _supports_box_drawing()

```python
def _supports_box_drawing() -> bool
```

Function to check if the terminal supports boxed characters.

**Returns**:

* _**bool**_ (`bool`): True if the terminal likely supports boxed characters, False otherwise.

---

### class TableFormatter

Class to format tabular data into strings for display in the console.

**Parameters**:

* _**headers**_ (`Mapping[str, int]`): Mapping of column names to their respective widths, in the format `{column_name: column_width}`.

**Raises**:

* `ValueError`: If `headers` is empty or not a valid mapping.

---

#### def _border(left, middle, right, line, new_line=True)

```python
def _border(self, left: str, middle: str, right: str, line: str, new_line: bool = True) -> str
```

Create a horizontal border for the table.

**Parameters**:

* _**left**_ (`str`): Character on the left side of the border.
* _**middle**_ (`str`): Character separator between columns.
* _**right**_ (`str`): Character on the right side of the border.
* _**line**_ (`str`): Character used to fill the border.
* _**new_line**_ (`bool`, optional): If True, adds a line break before the border (default is True).

**Returns**:

* _**border**_ (`str`): String representing the horizontal border.

---

#### def get_header()

```python
def get_header(self) -> str
```

Generate the table header, including the top border, column headings, and separator line.

**Returns**:

* _**header**_ (`str`): Formatted string of the table header.

---

#### def get_row(values)

```python
def get_row(self, values: Mapping[str, Union[str, int, float]]) -> str
```

Generate a formatted row for the table data.

**Parameters**:

* _**values**_ (`Mapping[str, Union[str, int, float]]`): Dictionary with values for each column, in the format `{column_name: value}`.

**Returns**:

* _**row**_ (`str`): Formatted string of the table row.

---

#### def get_bottom(new_line=False)

```python
def get_bottom(self, new_line: bool = False) -> str
```

Generate the table's bottom border.

**Parameters**:

* _**new_line**_ (`bool`, optional): If True, adds a line break before the border (default is False).

**Returns**:

* _**bottom**_ (`str`): Formatted string for the bottom border.

---

### class ProgressTable(TableFormatter)

Class to display a formatted table in the console to track the algorithm's progress.

**Parameters**:

* _**headers**_ (`Mapping[str, int]`): Mapping `{column_name: column_width}`.
* _**verbose**_ (`bool`, default=True): If False, prints nothing to the terminal.

**Raises**:

* `ValueError`: If `headers` is empty or not a valid mapping.

---

#### def _print_header()

```python
def _print_header(self) -> None
```

Print the table header.

---

#### def update(values)

```python
def update(self, values: Mapping[str, Union[str, int, float]]) -> None
```

Add a new row of values to the table.

**Parameters**:

* _**values**_ (`Mapping[str, Union[str, int, float]]`): Keys must match the columns defined in headers.

---

#### def finish()

```python
def finish(self) -> None
```

End the table display, printing the bottom border and total time.

---
