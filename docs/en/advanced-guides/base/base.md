# Base Class

Base class for parameter introspection compatible with the scikit-learn API.

## Base

Generic base class for models with a common interface.

Provides the `get_params` and `set_params` methods for compatibility with the scikit-learn API, allowing access to the model's public parameters.

### Function set_params(...)

```python
def set_params(self, **params)
```

Set the parameters of the instance. Ensures compatibility with scikit-learn functions.

**Parameters**:

* **params**: dict - Dictionary of parameters to set as attributes on the instance. Only public attributes (not starting with "_") are modified.

**Returns**:

* self: `Base` - Returns the instance itself after setting the parameters.

---

### Function get_params(...)

```python
def get_params(self, deep: bool = True) -> dict
```

Return a dictionary with the object's main parameters. Ensures compatibility with scikit-learn functions.

**Parameters**:

* **deep**: `bool` (default=True) - Ignored in this implementation but included for scikit-learn compatibility.

**Returns**:

* params: `dict` - Dictionary containing the object's attributes that do not start with "_".
