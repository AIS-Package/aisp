# BaseClusterer

Abstract base class for clustering algorithms.

This class defines the core interface for clustering models. It enforces
the implementation of the **`fit`** and **`predict`** methods in all derived classes.

---

### Method `fit(...)`

```python
@abstractmethod
def fit(self, X: Union[npt.NDArray, list], verbose: bool = True) -> BaseClusterer:
```

Fit the model to the training data.
This abstract method must be implemented by subclasses.

**Parameters**:

* **X** (`Union[npt.NDArray, list]`): Input data used for training the model.
* **verbose** (`bool`, default=True): Flag to enable or disable detailed output during training.

**Returns**:

* **self** (`BaseClusterer`): Instance of the class that implements this method.

**Implementation**:

* [AiNet](../../classes/Network%20Theory%20Algorithms/AiNet.md#method-fit)

---

### Method `predict(...)`

```python
@abstractmethod
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
```

Generate predictions based on the input data.
This abstract method must be implemented by subclasses.

**Parameters**:

* **X** (`Union[npt.NDArray, list]`): Input data for which predictions will be generated.

**Returns**:

* **predictions** (`npt.NDArray`): Predicted cluster labels for each input sample.

**Implementation**:

* [AiNet](../../classes/Network%20Theory%20Algorithms/AiNet.md#method-predict)

---

### Method `fit_predict(...)`

```python
def fit_predict(self, X: Union[npt.NDArray, list], verbose: bool = True) -> npt.NDArray:
```

Convenience method that combines `fit` and `predict` in a single call.

**Parameters**:

* **X** (`Union[npt.NDArray, list]`): Input data for which predictions will be generated.
* **verbose** (`bool`, default=True): Flag to enable or disable detailed output during training.

**Returns**:

* **predictions**: `npt.NDArray` - Predicted cluster labels for each input sample.
