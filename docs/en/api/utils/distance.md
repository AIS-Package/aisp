---
id: distance
sidebar_label: distance
keywords:
    - hamming
    - euclidean
    - cityblock
    - Manhattan
    - minkowski
    - distance
---

# distance

Utility functions for distance between arrays with numba decorators.

> **Module:** `aisp.utils.distance`  
> **Import:** `from aisp.utils import distance`

## Functions

### hamming

```python
@njit([(types.boolean[:], types.boolean[:])], cache=True)
def hamming(u: npt.NDArray[np.bool_], v: npt.NDArray[np.bool_]) -> float64:
    ...
```

Calculate the Hamming distance between two points.

$$
\frac{(x_1 \neq y_1) + (x_2 \neq y_2) + \cdots + (x_n \neq y_n)}{n}
$$

**Parameters**

| Name | Type                    | Default | Description                      |
|------|-------------------------|:-------:|----------------------------------|
| `u`  | `npt.NDArray[np.bool_]` |    -    | Coordinates of the first point.  |
| `v`  | `npt.NDArray[np.bool_]` |    -    | Coordinates of the second point. |

**Returns**

`float64` - Hamming distance between two points.

---

### euclidean

```python
@njit()
def euclidean(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float64:
    ...
```

Calculate the Euclidean distance between two points.

$$
\sqrt{(X_{1} - X_{1})^2 + (Y_{2} - Y_{2})^2 + \cdots + (Y_{n} - Y_{n})^2}
$$

**Parameters**

| Name | Type                   | Default | Description                      |
|------|------------------------|:-------:|----------------------------------|
| `u`  | `npt.NDArray[float64]` |    -    | Coordinates of the first point.  |
| `v`  | `npt.NDArray[float64]` |    -    | Coordinates of the second point. |

**Returns**

`float64` - Euclidean distance between two points.

---

### cityblock

```python
@njit()
def cityblock(u: npt.NDArray[float64], v: npt.NDArray[float64]) -> float64:
    ...
```

Calculate the Manhattan distance between two points.

$$
|X_{1} - Y_{1}| + |X_{2} - Y_{2}| + \cdots + |X_{n} - Y_{n}|
$$

**Parameters**

| Name | Type                   | Default | Description                      |
|------|------------------------|:-------:|----------------------------------|
| `u`  | `npt.NDArray[float64]` |    -    | Coordinates of the first point.  |
| `v`  | `npt.NDArray[float64]` |    -    | Coordinates of the second point. |

**Returns**

`float64` - Manhattan distance between two points.

---

### minkowski

```python
@njit()
def minkowski(
    u: npt.NDArray[float64],
    v: npt.NDArray[float64],
    p: float = 2.0
) -> float64:
    ...
```

Calculate the Minkowski distance between two points.

$$
(|X_{1} - Y_{1}|^p + |X_{2} - Y_{2}|^p + \cdots + |X_{n} - Y_{n}|^p)^\frac{1}{p}|
$$

**Parameters**

| Name | Type                   | Default | Description                                                    |
|------|------------------------|:-------:|----------------------------------------------------------------|
| `u`  | `npt.NDArray[float64]` |    -    | Coordinates of the first point.                                |
| `v`  | `npt.NDArray[float64]` |    -    | Coordinates of the second point.                               |
| `p`  | `float`                |  `2.0`  | The p parameter defines the type of distance to be calculated. |

:::note[Parameter `p`]

- p = 1: **Manhattan** distance - sum of absolute differences.
- p = 2: **Euclidean** distance - sum of squared differences (square root).
- p > 2: **Minkowski** distance with an increasing penalty as p increases.

:::

**Returns**

`float64` - Minkowski distance between two points.

---

### compute_metric_distance

```python
@njit([(types.float64[:], types.float64[:], types.int32, types.float64)], cache=True)
def compute_metric_distance(
    u: npt.NDArray[float64],
    v: npt.NDArray[float64],
    metric: int,
    p: float = 2.0
) -> float64:
    ...
```

Calculate the distance between two points by the chosen metric.

**Parameters**

| Name     | Type                   | Default | Description                                                                                |
|----------|------------------------|:-------:|--------------------------------------------------------------------------------------------|
| `u`      | `npt.NDArray[float64]` |    -    | Coordinates of the first point.                                                            |
| `v`      | `npt.NDArray[float64]` |    -    | Coordinates of the second point.                                                           |
| `metric` | `int`                  |    -    | Distance metric to be used. Available options: 0 (Euclidean), 1 (Manhattan), 2 (Minkowski) |
| `p`      | `float`                |  `2.0`  | The p parameter defines the type of distance to be calculated.                             |

**Raises**

{{ Raise }} - {{ Description }}.

**Returns**

`float64` - Distance between the two points with the selected metric.

---

### min_distance_to_class_vectors

```python
@njit([(types.float64[:, :], types.float64[:], types.int32, types.float64)], cache=True)
def min_distance_to_class_vectors(
    x_class: npt.NDArray[float64],
    vector_x: npt.NDArray[float64],
    metric: int,
    p: float = 2.0,
) -> float:
    ...
```

Calculate the minimum distance between an input vector and the vectors of a class.

**Parameters**

| Name       | Type                   | Default | Description                                                                                                       |
|------------|------------------------|:-------:|-------------------------------------------------------------------------------------------------------------------|
| `x_class`  | `npt.NDArray[float64]` |    -    | Array containing the class vectors to be compared with the input vector. Expected shape: (n_samples, n_features). |
| `vector_x` | `npt.NDArray[float64]` |    -    | Vector to be compared with the class vectors. Expected shape: (n_features,).                                      |
| `metric`   | `int`                  |    -    | Distance metric to be used. Available options: 0 ("euclidean"), 1 ("manhattan"), 2 ("minkowski") or ("hamming")   |
| `p`        | `float`                |  `2.0`  | Parameter for the Minkowski distance (used only if `metric` is "minkowski").                                      |

**Returns**

**min_distance**: `float` - The minimum distance calculated between the input vector and the class vectors. Returns -1.0 if the input dimensions are incompatible.

---

### get_metric_code

```python
def get_metric_code(metric: str) -> int:
    ...
```

Get the numeric code associated with a distance metric.

**Parameters**

| Name     | Type  | Default | Description                                                                            |
|----------|-------|:-------:|----------------------------------------------------------------------------------------|
| `metric` | `str` |    -    | Name of the metric. Can be `"euclidean"`, `"manhattan"`, `"minkowski"` or `"hamming"`. |

**Raises**

* `ValueError` - If the metric provided is not supported.

**Returns**

`int` - Numeric code corresponding to the metric.
