Base class for classification algorithm.

# BaseClassifier

Base class for classification algorithms, defining the abstract methods ``fit`` and ``predict``, and implementing the ``get_params`` method.

### Method `score(...)`

```python
def score(self, X: npt.NDArray, y: list) -> float
```

Score function calculates forecast accuracy.

This function performs the prediction of X and checks how many elements are equal between vector y and y_predicted.
This function was added for compatibility with some scikit-learn functions.

**Parameters:**

* **X** (`npt.NDArray`): Feature set with shape (n_samples, n_features).
* **y** (`list`): True values with shape (n_samples,).

**Returns**:

* accuracy: ``float`` The accuracy of the model.

---

### Method `_slice_index_list_by_class(...)`

The function ``_slice_index_list_by_class(...)```, separates the indices of the lines according to the output class, to go through the sample array, only in the positions that the output is the class that is being trained:

```python
def _slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Returns a dictionary with the classes as key and the indices in ``X`` of the samples.

---

## Abstract methods

### Method `fit(...)`

```python
@abstractmethod
def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True) -> BaseClassifier:
```

Fit the model to the training data.

Implementation:

* [RNSA](../../classes/Negative%20Selection/RNSA.md#method-fit)
* [BNSA](../../classes/Negative%20Selection/BNSA.md#method-fit)
* [AIRS](../../classes/Clonal%20Selection%20Algorithms/AIRS.md#method-fit)

### Method `predict(...)`

```python
@abstractmethod
def predict(self, X) -> Optional[npt.NDArray]:
```

Performs label prediction for the given data.

Implementation:

* [RNSA](../../classes/Negative%20Selection/RNSA.md#method-predict)
* [BNSA](../../classes/Negative%20Selection/BNSA.md#method-predict)
* [AIRS](../../classes/Clonal%20Selection%20Algorithms/AIRS.md#method-predict)
