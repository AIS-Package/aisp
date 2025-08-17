# BaseAiNet(BaseClusterer, ABC)

Base class for Network Theory algorithms based on AiNet.

The base class contains functions that are used by multiple classes in the AiNet package and
are considered essential for proper functioning of clustering algorithms based on immune network theory.

---

### def _check_and_raise_exceptions_fit(...)

Verify the fit parameters and throw exceptions if the verification is not successful.

```python
@staticmethod
def _check_and_raise_exceptions_fit(X: npt.NDArray)
```

**Parameters**:

* ***X*** (`npt.NDArray`): Training array, containing the samples and their characteristics, \[`N samples` (rows)]\[`N features` (columns)].

**Raises**:

* `TypeError`: If X is not an ndarray or list.

---

### def _check_and_raise_exceptions_predict(...)

Verify the predict parameters and throw exceptions if the verification is not successful.

```python
@staticmethod
def _check_and_raise_exceptions_predict(
    X: npt.NDArray,
    expected: int = 0,
    feature_type: FeatureType = "continuous-features"
) -> None
```

**Parameters**:

* ***X*** (`npt.NDArray`): Input array for prediction, containing the samples and their characteristics, \[`N samples` (rows)]\[`N features` (columns)].
* ***expected*** (`int`, default=0): Expected number of features per sample (columns in X).
* ***feature_type*** (`FeatureType`, default="continuous-features"): Specifies the type of features: "continuous-features", "binary-features", or "ranged-features".

**Raises**:

* `TypeError`: If X is not an ndarray or list.
* `FeatureDimensionMismatch`: If the number of features in X does not match the expected number.
* `ValueError`: If feature_type is "binary-features" and X contains values other than 0 and 1.

---

### def _generate_random_antibodies(...)

Generate a random antibody population.

```python
@staticmethod
def _generate_random_antibodies(
    n_samples: int,
    n_features: int,
    feature_type: FeatureType = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray
```

**Parameters**:

* ***n_samples*** (`int`): Number of antibodies (samples) to generate.
* ***n_features*** (`int`): Number of features (dimensions) for each antibody.
* ***feature_type*** (`FeatureType`, default="continuous-features"): Specifies the type of features: "continuous-features", "binary-features", or "ranged-features".
* ***bounds*** (`Optional[npt.NDArray[np.float64]]`): Array of shape (n_features, 2) with min and max per dimension (used only for ranged features).

**Returns**:

* `npt.NDArray`: Array of shape (n_samples, n_features) containing the generated antibodies.
  Data type depends on the feature_type (float for continuous/ranged, bool for binary).

**Raises**:

* `ValueError`: If n_features <= 0.

---

### Abstract Methods

#### fit(...)

```python
def fit(self, X: npt.NDArray, verbose: bool = True) -> "BaseAiNet"
```

Train the AiNet clustering model.
Must be implemented by subclasses.

**Parameters**:

* ***X*** (`npt.NDArray`): Input data used for training the model.
* ***verbose*** (`bool`, default=True): Flag to enable or disable detailed output during training.

**Returns**:

* `BaseAiNet`: Instance of the class implementing the method.

#### predict(...)

```python
def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]
```

Generate cluster predictions for the input data.
Must be implemented by subclasses.

**Parameters**:

* ***X*** (`npt.NDArray`): Input data for which predictions will be generated.

**Returns**:

* `Optional[npt.NDArray]`: Predicted cluster labels for each input sample, or None if prediction is not possible.

#### fit_predict(...)

```python
def fit_predict(self, X: npt.NDArray, verbose: bool = True) -> Optional[npt.NDArray]
```

Convenience method combining `fit` and `predict` in a single call.

**Parameters**:

* ***X*** (`npt.NDArray`): Input data for which predictions will be generated.
* ***verbose*** (`bool`, default=True): Flag to enable or disable detailed output during training.

**Returns**:

* `Optional[npt.NDArray]`: Predicted cluster labels for each input sample, or None if prediction is not possible.

**Implementation**:

* Calls `fit` internally, then `predict`.
* Used by AiNet-based algorithms in Immune Network Theory.
