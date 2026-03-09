---
id: multiclass
sidebar_label: multiclass
keywords:
    - k-nearest neighbors
    - samples
    - slice
    - multiclass
---

# multiclass

Utility functions for handling classes with multiple categories.

> **Module:** `aisp.utils.multiclass`  
> **Import:** `from aisp.utils import multiclass`

## Functions

### slice_index_list_by_class

```python
def slice_index_list_by_class(classes: Optional[Union[npt.NDArray, list]], y: npt.NDArray) -> dict:
    ...
```

Separate indices of samples by class for targeted iteration.

**Parameters**

| Name      | Type                                 | Default | Description                                                                         |
|-----------|--------------------------------------|:-------:|-------------------------------------------------------------------------------------|
| `classes` | `Optional[Union[npt.NDArray, list]]` |    -    | list with unique classes.                                                           |
| `y`       | `npt.NDArray`                        |    -    | Receives a `y` (`n_samples`) array with the output classes of the `X` sample array. |

**Returns**

**position_samples**: `dict` - A dictionary with the list of array positions(``y``), with the classes as key.

**Example**

```python
import numpy as np
from aisp.utils.multiclass import slice_index_list_by_class

labels = ['a', 'b', 'c']
y = np.array(['a', 'c', 'b', 'a', 'c', 'b'])
slice_index_list_by_class(labels, y)
```

---

### predict_knn_affinity

```python
def predict_knn_affinity(
    X: npt.NDArray,
    k: int,
    all_cell_vectors: List[Tuple[Union[str, int], npt.NDArray]],
    affinity_func: Callable[[npt.NDArray, npt.NDArray], float]
) -> npt.NDArray:
    ...
```

Predict classes using k-nearest neighbors and trained cells.

**Parameters**

| Name               | Type                                          | Default | Description                                                    |
|--------------------|-----------------------------------------------|:-------:|----------------------------------------------------------------|
| `X`                | `npt.NDArray`                                 |    -    | Input data to be classified.                                   |
| `k`                | `int`                                         |    -    | Number of nearest neighbors to consider for prediction.        |
| `all_cell_vectors` | `List[Tuple[Union[str, int], npt.NDArray]]`   |    -    | List of tuples (class_name, cell(np.ndarray)).                 |
| `affinity_func`    | `Callable[[npt.NDArray, npt.NDArray], float]` |    -    | Function that takes two vectors and returns an affinity value. |

**Returns**

**predicted_labels**: `npt.NDArray` - Array of predicted labels for each sample in X, based on the k nearest neighbors.
