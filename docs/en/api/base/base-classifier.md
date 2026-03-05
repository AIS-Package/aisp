---
id: base-classifier
sidebar_label: BaseClassifier
keywords:
    - base
    - classifier
    - classifier interface
    - accuracy score
    - fit
    - predict
tags:
    - classifier
    - classification
---

# BaseClassifier

Abstract base class for classification algorithms.

> **Module:** `aisp.base`  
> **Import:** `from aisp.base import BaseClassifier`

---

## Overview

This class defines the core interface for classification models. It enforces the implementation of the `fit`
and `predict` methods in all derived classes, and provides a default implementation of the `score` method.

Use cases:

- Abstract base class for extending classification algorithm classes.

---

## Attributes

| Name          | Type                    | Default | Description                              |
|---------------|-------------------------|:-------:|------------------------------------------|
| `classes`     | `Optional[npt.NDArray]` | `None`  | Class labels identified during training. |

---

## Abstract Methods

### fit

```python
@abstractmethod
def fit(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list],
    verbose: bool = True
) -> BaseClassifier:
    ...
```

Train the model using the input data X and corresponding labels y.  
This abstract method is implemented by the class that inherits it.

**Parameters**

| Name      | Type                       | Default | Description                                                |
|-----------|----------------------------|:-------:|------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |    -    | Input data used for training the model.                    |
| `y`       | `Union[npt.NDArray, list]` |    -    | Corresponding labels or target values for the input data.  |
| `verbose` | `bool`                     | `True`  | Flag to enable or disable detailed output during training. |

**Returns**

``BaseClassifier`` - Returns the instance of the class that implements this method.

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

| Name | Type                       | Default | Description                                         |
|------|----------------------------|:-------:|-----------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |    -    | Input data for which predictions will be generated. |

**Returns**

`npt.NDArray` - Predicted values for each input sample.

---

## Public Methods

### score

```python
def score(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list]
) -> float:
    ...
```

Score function calculates forecast accuracy.

This function performs the prediction of X and checks how many elements are equal between vector y and y_predicted.
This function was added for compatibility with some scikit-learn functions.

**Parameters**

| Name      | Type   | Default | Description                   |
|-----------|--------|:-------:|-------------------------------|
| `X` | `Union[npt.NDArray, list]` |    -    | Feature set with shape (n_samples, n_features). |
| `y` | `Union[npt.NDArray, list]` |    -    | True values with shape (n_samples,). |

**Returns**

`float` - The accuracy of the model.

