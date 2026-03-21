---
id: sanitizers
sidebar_label: sanitizers
keywords:
    - sanitize
---

# sanitizers

Utility functions for validation and treatment of parameters.

> **Module:** `aisp.utils.sanitizers`  
> **Import:** `from aisp.utils import sanitizers`

## Functions

### sanitize_choice

```python
def sanitize_choice(value: T, valid_choices: Iterable[T], default: T) -> T:
    ...
```

Value if present in the set of valid choices; otherwise, the default value.

**Parameters**

| Name            | Type          | Default | Description                                                            |
|-----------------|---------------|:-------:|------------------------------------------------------------------------|
| `value`         | `T`           |    -    | The value to be checked.                                               |
| `valid_choices` | `Iterable[T]` |    -    | A collection of valid choices.                                         |
| `default`       | `T`           |    -    | The default value to be returned if 'value' is not in 'valid_choices'. |

**Returns**

| Type | Description                                               |
|------|-----------------------------------------------------------|
| `T`  | The original value if valid, or the default value if not. |

---

### sanitize_param

```python
def sanitize_param(value: T, default: T, condition: Callable[[T], bool]) -> T:
    ...
```

Value if it satisfies the specified condition; otherwise, the default value.

**Parameters**

| Name        | Type                  | Default | Description                                                                             |
|-------------|-----------------------|:-------:|-----------------------------------------------------------------------------------------|
| `value`     | `T`                   |    -    | The value to be checked.                                                                |
| `default`   | `T`                   |    -    | The default value to be returned if the condition is not satisfied.                     |
| `condition` | `Callable[[T], bool]` |    -    | A function that takes a value and returns a boolean, determining if the value is valid. |

**Returns**

| Type | Description                                                                    |
|------|--------------------------------------------------------------------------------|
| `T`  | The original value if the condition is satisfied, or the default value if not. |

---

### sanitize_seed

```python
def sanitize_seed(seed: Any) -> Optional[int]:
    ...
```

Seed if it is a non-negative integer; otherwise, None.

**Parameters**

| Name   | Type  | Default | Description                     |
|--------|-------|:-------:|---------------------------------|
| `seed` | `Any` |    -    | The seed value to be validated. |

**Returns**

| Type            | Description                                                                  |
|-----------------|------------------------------------------------------------------------------|
| `Optional[int]` | The original seed if it is a non-negative integer, or None if it is invalid. |


**seed**: `Optional[int]` - The original seed if it is a non-negative integer, or None if it is invalid.

---

### sanitize_bounds

```python
def sanitize_bounds(
    bounds: Any, problem_size: int
) -> Dict[str, npt.NDArray[np.float64]]:
    ...
```

Validate and normalize feature bounds.

**Parameters**

| Name           | Type  | Default | Description                                                                                                  |
|----------------|-------|:-------:|--------------------------------------------------------------------------------------------------------------|
| `bounds`       | `Any` |    -    | The input bounds, which must be either None or a dictionary with 'low' and 'high' keys.                      |
| `problem_size` | `int` |    -    | The expected length for the normalized bounds lists, corresponding to the number of features in the problem. |

**Returns**

| Type              | Description                                                               |
|-------------------|---------------------------------------------------------------------------|
| `Dict[str, list]` | Dictionary `{'low': [low_1, ..., low_N], 'high': [high_1, ..., high_N]}`. |

**Raises**

| Exception    | Description                                                                            |
|--------------|----------------------------------------------------------------------------------------|
| `TypeError`  | If `bounds` is not None and not a dict with 'low'/'high', or if items are non-numeric. |
| `ValueError` | If provided iterables have the wrong length.                                           |
