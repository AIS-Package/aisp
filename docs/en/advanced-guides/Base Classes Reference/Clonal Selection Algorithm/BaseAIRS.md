# BaseAIRS(BaseClassifier, ABC)

Base class for algorithm **AIRS**.

The base class contains functions that are used by more than one class in the package, and
therefore are considered essential for the overall functioning of the system.

---

### def _check_and_raise_exceptions_fit(...):

 Verify the fit parameters and throw exceptions if the verification is not successful.

```python
@staticmethod
def _check_and_raise_exceptions_fit(
    X: npt.NDArray = None,
    y: npt.NDArray = None,
    algorithm: Literal[
        "continuous-features", "binary-features"
    ] = "continuous-features"
):
```


**Parameters**:
* ***X*** (``npt.NDArray``): Training array, containing the samples and their characteristics, [``N samples`` (rows)][``N features`` (columns)].
* ***y*** (``npt.NDArray``): Array of target classes of ``X`` with [``N samples`` (lines)].
* ***algorithm*** (``Literal["continuous-features", "binary-features"], optional``): Specifies the type of algorithm to use, depending 
on whether the input data has continuous or binary features.

**Raises**
* `TypeError`:
    If X or y are not ndarrays or have incompatible shapes.
* `ValueError`
    If _class_ is BNSA and X contains values that are not composed only of 0 and 1.
---

---

### def _check_and_raise_exceptions_fit(...):

 Verify the fit parameters and throw exceptions if the verification is not successful.

```python
@staticmethod
def _check_and_raise_exceptions_predict(
    X: npt.NDArray = None,
    expected: int = 0,
    algorithm: Literal[
        "continuous-features", "binary-features"
    ] = "continuous-features"
) -> None:
```


**Parameters**:
* ***X*** (``npt.NDArray``): Training array, containing the samples and their characteristics, [``N samples`` (rows)][``N features`` (columns)].
* ***expected*** (``int``):  Expected number of features per sample (columns in X).
* ***algorithm*** (``Literal["continuous-features", "binary-features"], optional``): Specifies the type of algorithm to use, depending 
on whether the input data has continuous or binary features.

**Raises**
* ``TypeError``
    If X is not a ndarray or list.
* `FeatureDimensionMismatch`
    If the number of features in X does not match the expected number.
* `ValueError`
    If algorithm is binary-features and X contains values that are not composed only of 0 and 1.
---