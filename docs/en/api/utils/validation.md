---
id: validation
sidebar_label: validation
keywords:
    - validation
---

# module

Contains functions responsible for validating data types.

> **Module:** `aisp.utils.validation`  
> **Import:** `from aisp.utils import validation`

## Functions

### detect_vector_data_type

```python
def detect_vector_data_type(vector: npt.NDArray) -> FeatureType:
    ...
```

Detect the type of data in a vector.

The function detects if the vector contains data of type:
- Binary features: boolean values or integers restricted to 0 and 1.
- Continuous features: floating-point values in the normalized range [0.0, 1.0].
- Ranged features: floating-point values outside the normalized range.

**Parameters**

| Name     | Type          | Default | Description                                    |
|----------|---------------|:-------:|------------------------------------------------|
| `vector` | `npt.NDArray` |    -    | An array containing the data to be classified. |

**Raises**

* `UnsupportedTypeError` -  If the data type of the vector is not supported by the function.

**Returns**

[`FeatureType`](./types.md#featuretype) - The data type of the vector: "binary-features", "continuous-features", or "ranged-features".

---

### check_array_type

```python
def check_array_type(x, name: str = "X") -> npt.NDArray:
    ...
```

Ensure X is a numpy array. Convert from list if needed.

**Parameters**

| Name   | Type  | Default | Description                                                                              |
|--------|-------|:-------:|------------------------------------------------------------------------------------------|
| `x`    | `Any` |    -    | Array, containing the samples and their characteristics, Shape: (n_samples, n_features). |
| `name` | `str` |  `'X'`  | Variable name used in error messages.                                                    |

**Raises**

* `TypeError` - If X or y are not ndarrays or a list.

**Returns**

`npt.NDArray` - The converted or validated array

---

### check_shape_match

```python
def check_shape_match(x: npt.NDArray, y: npt.NDArray):
    ...
```

Ensure X and y have compatible first dimensions.

**Parameters**

| Name | Type          | Default | Description                                                                              |
|------|---------------|:-------:|------------------------------------------------------------------------------------------|
| `x`  | `npt.NDArray` |    -    | Array, containing the samples and their characteristics, Shape: (n_samples, n_features). |
| `y`  | `npt.NDArray` |    -    | Array of target classes of `x` with (`n_samples`).                                       |

**Raises**

* `TypeError` - If x or y have incompatible shapes.

---

### check_feature_dimension

```python
def check_feature_dimension(x: npt.NDArray, expected: int):
    ...
```

Ensure X has the expected number of features.

**Parameters**

| Name       | Type          | Default | Description                                                                                                   |
|------------|---------------|:-------:|---------------------------------------------------------------------------------------------------------------|
| `x`        | `npt.NDArray` |    -    | Input array for prediction, containing the samples and their characteristics, Shape: (n_samples, n_features). |
| `expected` | `int`         |    -    | Expected number of features per sample (columns in X).                                                        |

**Raises**

* `FeatureDimensionMismatch` - If the number of features in X does not match the expected number.

---

### check_binary_array

```python
def check_binary_array(x: npt.NDArray):
    ...
```

Ensure X contains only 0 and 1.

**Parameters**

| Name | Type          | Default | Description                    |
|------|---------------|:-------:|--------------------------------|
| `x`  | `npt.NDArray` |    -    | Array, containing the samples. |

**Raises**

* `ValueError` - If the array contains values other than 0 and 1.

---

### check_value_range

```python
def check_value_range(
    x: npt.NDArray,
    name: str = 'X',
    min_value: float = 0.0,
    max_value: float = 1.0
) -> None:
    ...
```

Ensure all values in the x array fall within a range.

**Parameters**

| Name        | Type          | Default | Description                     |
|-------------|---------------|:-------:|---------------------------------|
| `x`         | `npt.NDArray` |    -    | Array, containing the samples.  |
| `name`      | `str`         |  `'X'`  | Name used in the error message. |
| `min_value` | `float`       |  `0.0`  | Minimum allowed value.          |
| `max_value` | `float`       |  `1.0`  | Maximum allowed value.          |

**Raises**

* `ValueError` - If the array fall outside the interval (min_value, max_value).
