---
id: exceptions
sidebar_label: aisp.exceptions
keywords:
    - exceptions
    - raises
    - warnings
---

# aisp.exceptions

Custom warnings and errors.

> **Module:** `aisp.exceptions`  
> **Import:** `from aisp import exceptions`

## Exception classes

### MaxDiscardsReachedError

```python
class MaxDiscardsReachedError(Exception):
    ...
```

Exception thrown when the maximum number of detector discards is reached.

**Parameters**

| Name          | Type            | Default | Description                                                    |
|---------------|-----------------|:-------:|----------------------------------------------------------------|
| `object_name` | `str`           |    -    | The name of the instantiated class that throws the exceptions. |
| `message`     | `Optional[str]` | `None`  | Custom message to display.                                     |

---

### FeatureDimensionMismatch

```python
class FeatureDimensionMismatch(Exception):
    ...
```

Exception raised when the number of input features does not match the expected number.  
This exception is triggered during prediction if the input features' dimension is incorrect.

**Parameters**

| Name            | Type            | Default | Description                                         |
|-----------------|-----------------|:-------:|-----------------------------------------------------|
| `expected`      | `int`           |    -    | The expected number of features                     |
| `received`      | `int`           |    -    | The actual number of features received.             |
| `variable_name` | `Optional[str]` | `None`  | The name of the variable that caused this mismatch. |

---
### UnsupportedTypeError

```python
class UnsupportedTypeError(Exception):
    ...
```

Exception raised when the input vector type is not supported.  
This exception is thrown when the vector data type does not match any of the supported.

**Parameters**

| Name      | Type            | Default | Description                |
|-----------|-----------------|:-------:|----------------------------|
| `message` | `Optional[str]` | `None`  | Custom message to display. |

---
### ModelNotFittedError

```python
class ModelNotFittedError(Exception):
    ...
```

Exception raised when a method is called before the model has been fit.  
This exception is thrown when the  model instance is being used without first training
it via the `fit` method.

**Parameters**

| Name          | Type            | Default | Description                                                    |
|---------------|-----------------|:-------:|----------------------------------------------------------------|
| `object_name` | `str`           |    -    | The name of the instantiated class that throws the exceptions. |
| `message`     | `Optional[str]` | `None`  | Custom message to display.                                     |
