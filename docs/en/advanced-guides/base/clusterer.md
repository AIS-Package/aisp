# BaseClusterer

Abstract base class for clustering algorithms.

This class defines the core interface for clustering models. It enforces
the implementation of the **`fit`** and **`predict`** methods in all derived classes,
and provides a default implementation for **`fit_predict`** and **`get_params`**.

---

### Function fit(...)

```python
@abstractmethod
def fit(self, X: npt.NDArray, verbose: bool = True) -> BaseClusterer:
```

Fit the model to the training data.
This abstract method must be implemented by subclasses.

**Parameters**:

* ***X***: `npt.NDArray` - Input data used for training the model.
* ***verbose***: `bool`, default=True - Flag to enable or disable detailed output during training.

**Returns**:

* ***self***: `BaseClusterer` - Instance of the class that implements this method.

**Implementation**:

* [AiNet](../../classes/Network%20Theory%20Algorithms/AiNet.md#function-fit)

---

### Function predict(...)

```python
@abstractmethod
def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]:
```

Generate predictions based on the input data.
This abstract method must be implemented by subclasses.

**Parameters**:

* ***X***: `npt.NDArray` - Input data for which predictions will be generated.

**Returns**:

* ***predictions***: `Optional[npt.NDArray]` - Predicted cluster labels for each input sample, or `None` if prediction is not possible.

**Implementation**:

* [AiNet](../../classes/Network%20Theory%20Algorithms/AiNet.md#function-predict)

---

### Function fit_predict(...)

```python
def fit_predict(self, X: npt.NDArray, verbose: bool = True) -> Optional[npt.NDArray]
```

Convenience method that combines `fit` and `predict` in a single call.

**Parameters**:

* ***X***: `npt.NDArray` - Input data for which predictions will be generated.
* ***verbose***: `bool`, default=True - Flag to enable or disable detailed output during training.

**Returns**:

* ***predictions***: `Optional[npt.NDArray]` - Predicted cluster labels for each input sample, or `None` if prediction is not possible.
