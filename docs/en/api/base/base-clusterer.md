---
id: base-clusterer
sidebar_label: BaseClusterer
keywords:
    - base
    - clusterer
    - clusterer interface
    - cluster labels
    - fit
    - predict
    - fit_predict
tags:
    - clusterer
    - clustering
---

# BaseClusterer

Abstract base class for clustering algorithms.

> **Module:** `aisp.base`  
> **Import:** `from aisp.base import BaseClusterer`

---

## Overview

This class defines the core interface for clustering models. It enforces the implementation of the `fit` and
`predict` methods in all derived classes, and provides a default implementation for `fit_predict` and `get_params`.

Use cases:

- Abstract base class for extending clustering algorithm classes.

---

## Attributes

| Name     | Type                    | Default | Description                                             |
|----------|-------------------------|:-------:|---------------------------------------------------------|
| `labels` | `Optional[npt.NDArray]` | `None`  | Labels for the clusters generated during model fitting. |

---

## Abstract  Methods

### fit

```python
@abstractmethod
def fit(self, X: Union[npt.NDArray, list], verbose: bool = True) -> BaseClusterer:
    ...
```

Train the model using the input data X.  
This abstract method is implemented by the class that inherits it.

**Parameters**

| Name      | Type                       | Default | Description                                                                      |
|-----------|----------------------------|:-------:|----------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |    -    | Training input samples. Each row corresponds to a samples and column to feature. |
| `verbose` | `bool`                     | `True`  | If True, prints training progress information.                                   |

**Returns**

| Type   | Description                                                    |
|--------|----------------------------------------------------------------|
| `Self` | Returns the instance of the class that implements this method. |

---

### predict

```python
@abstractmethod
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
    ...
```

Generate predictions based on the input data X.  
This abstract method is implemented by the class that inherits it.

**Parameters**

| Name | Type                       | Default | Description                                                                |
|------|----------------------------|:-------:|----------------------------------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |    -    | Input samples. Must have the same number of features used during training. |

**Returns**

| Type          | Description                                     |
|---------------|-------------------------------------------------|
| `npt.NDArray` | Predicted cluster labels for each input sample. |

---

## Public Methods

### fit_predict

```python
def fit_predict(self, X: Union[npt.NDArray, list], verbose: bool = True) -> npt.NDArray:
    ...
```

Fit the clustering model to the data and return cluster labels.

This is a convenience method that combines `fit` and `predict` into a single call.

**Parameters**

| Name      | Type                       | Default | Description                                                                       |
|-----------|----------------------------|:-------:|-----------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |    -    | Training input samples. Each row corresponds to a samples and column to feature.  |
| `verbose` | `bool`                     | `True`  | If True, prints training progress information.                                    |

**Returns**

| Type          | Description                                     |
|---------------|-------------------------------------------------|
| `npt.NDArray` | Predicted cluster labels for each input sample. |
