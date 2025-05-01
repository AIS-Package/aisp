# BaseNSA class

The ``_Base`` class contains utility functions with the ``protected`` modifier that can be inherited by various classes for ease of use. It includes functions for distance calculation, data separation to improve training and prediction efficiency, accuracy measurement and other functions.

---

## Functions

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

## Protected Functions:

---

### Function _distance(...):

```python
def _distance(self, u: npt.NDArray, v: npt.NDArray)
```

Function to calculate the distance between two points by the chosen ``metric``.

**Parameters**:
* ***u*** (``npt.NDArray``): Coordinates of the first point.
* ***v*** (``npt.NDArray``): Coordinates of the second point.

**returns**:
* Distance (``double``) between the two points.

---

### Function _check_and_raise_exceptions_fit(...):
```python
def _check_and_raise_exceptions_fit(self, X: npt.NDArray = None, y: npt.NDArray = None, _class_: Literal['RNSA', 'BNSA'] = 'RNSA')
```
Function responsible for verifying fit function parameters and throwing exceptions if the verification is not successful.

**Parameters**:
* ***X*** (``npt.NDArray``): Training array, containing the samples and their characteristics, [``N samples`` (rows)][``N features`` (columns)].
* ***y*** (``npt.NDArray``): Array of target classes of ``X`` with [``N samples`` (lines)].
* ***_class_*** (Literal[RNSA, BNSA], optional): Current class. Defaults to 'RNSA'.

---

## Abstract methods

### Function fit(...)

```python
def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True)
```

Fit the model to the training data.

Implementation:

- [RNSA](../../classes/Negative%20Selection/RNSA.md#function-fit)
- [BNSA](../../classes/Negative%20Selection/BNSA.md#function-fit)



### Function predict(...)

```python
def predict(self, X) -> Optional[npt.NDArray]:
```

Performs label prediction for the given data.

Implementation:

- [RNSA](../../classes/Negative%20Selection/RNSA.md#function-predict)
- [BNSA](../../classes/Negative%20Selection/BNSA.md#function-predict)


# Detector class

Represents a non-self detector of the RNSA class.

Attributes
----------
* ***position*** (``np.ndarray``): Detector feature vector.
* ***radius*** (``float, optional``): Detector radius, used in the V-detector algorithm.

