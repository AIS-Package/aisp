---
id: metrics
sidebar_label: metrics
keywords:
    - accuracy
    - score
---

# metrics

Utility functions for measuring accuracy and performance.

> **Module:** `aisp.utils.metrics`  
> **Import:** `from aisp.utils import metrics`

## Functions

### accuracy_score

```python
def accuracy_score(
    y_true: Union[npt.NDArray, list],
    y_pred: Union[npt.NDArray, list]
) -> float:
    ...
```

Calculate the accuracy score based on true and predicted labels.

**Parameters**

| Name     | Type                       | Default | Description                                                                   |
|----------|----------------------------|:-------:|-------------------------------------------------------------------------------|
| `y_true` | `Union[npt.NDArray, list]` |    -    | Ground truth (correct) labels. Expected to be of the same length as `y_pred`. |
| `y_pred` | `Union[npt.NDArray, list]` |    -    | Predicted labels. Expected to be of the same length as `y_true`.              |

**Raises**

* `ValueError` - If `y_true` or `y_pred` are empty or if they do not have the same length.

**Returns**

**accuracy**: `float` - The ratio of correct predictions to the total number of predictions.
