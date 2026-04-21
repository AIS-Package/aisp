---
id: types
sidebar_label: types
keywords:
    - types
    - typing
    - aliases
    - type hints
---

# types

Defines type aliases used throughout the project to improve readability.

> **Module:** `aisp.utils.types`  
> **Import:** `from aisp.utils import types`

## Type aliases

### FeatureType

```python
FeatureType: TypeAlias = Literal[
    "binary-features", "continuous-features", "ranged-features"
]
```

Type of input features:

- `"binary-features"`: values like 0 or 1.
- `"continuous-features"`: numeric continuous values.
- `"ranged-features"`: values defined by intervals.

---

### FeatureTypeAll

```python
FeatureTypeAll: TypeAlias = Union[FeatureType, Literal["permutation-features"]]
```

Same as ``FeatureType``, plus:

- `"permutation-features"`: values represented as permutation.

---

### MetricType

```python
MetricType: TypeAlias = Literal["manhattan", "minkowski", "euclidean"]
```

Distance metric used in calculations:

- `"manhattan"`: the Manhattan distance between two points
- `"minkowski"`: the Minkowski distance between two points.
- `"euclidean"`: the Euclidean distance between two points.

---