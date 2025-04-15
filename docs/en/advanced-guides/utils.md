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