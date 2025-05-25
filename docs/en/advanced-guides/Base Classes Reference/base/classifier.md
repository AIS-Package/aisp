Base class for classification algorithm.

# BaseClassifier

Base class for classification algorithms, defining the abstract methods ``fit`` and ``predict``, and implementing the ``get_params`` method.


### Function score(...)

```python
def score(self, X: npt.NDArray, y: list) -> float
```
Score function calculates forecast accuracy.

This function performs the prediction of X and checks how many elements are equal between vector y and y_predicted. 
This function was added for compatibility with some scikit-learn functions.

**Parameters**:
+ ***X***: ``np.ndarray``
    Feature set with shape (n_samples, n_features).
+ ***y***: ``np.ndarray``
    True values with shape (n_samples,).

**Returns**:

+ accuracy: ``float`` The accuracy of the model.

---

### Function _slice_index_list_by_class(...):

The function ``_slice_index_list_by_class(...)``, separates the indices of the lines according to the output class, to go through the sample array, only in the positions that the output is the class that is being trained:

```python
def _slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Returns a dictionary with the classes as key and the indices in ``X`` of the samples.

---

### Function get_params(...)

```python
def get_params(self, deep: bool = True) -> dict:
```
The get_params function Returns a dictionary with the object's main parameters.

This function is required to ensure compatibility with scikit-learn functions.

---

## Abstract methods

### Function fit(...)

```python
def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True)
```

Fit the model to the training data.

Implementation:

- [RNSA](../../../classes/Negative%20Selection/RNSA.md#function-fit)
- [BNSA](../../../classes/Negative%20Selection/BNSA.md#function-fit)
- [AIRS](../../../classes/Clonal%20Selection%20Algorithms/AIRS.md#function-fit)



### Function predict(...)

```python
def predict(self, X) -> Optional[npt.NDArray]:
```

Performs label prediction for the given data.

Implementation:

- [RNSA](../../../classes/Negative%20Selection/RNSA.md#function-predict)
- [BNSA](../../../classes/Negative%20Selection/BNSA.md#function-predict)
- [AIRS](../../../classes/Clonal%20Selection%20Algorithms/AIRS.md#function-predict)

# Detector class

Represents a non-self detector of the RNSA class.

Attributes
----------
* ***position*** (``np.ndarray``): Detector feature vector.
* ***radius*** (``float, optional``): Detector radius, used in the V-detector algorithm.

