**Negative Selection**

The functions perform detector checks and utilize Numba decorators for Just-In-Time compilation

### Function check_detector_bnsa_validity(...)

```python
def check_detector_bnsa_validity(
    x_class: npt.NDArray,
    vector_x: npt.NDArray,
    aff_thresh: float
) -> bool:
```

Checks the validity of a candidate detector (vector_x) against samples from a class (x_class) using the Hamming distance. A detector is considered INVALID if its distance to any sample in ``x_class`` is less than or equal to ``aff_thresh``.

**Parameters**:

* x_class (``npt.NDArray``): Array containing the class samples. Expected shape: (n_samples, n_features).
* vector_x (``npt.NDArray``): Array representing the detector. Expected shape: (n_features,).
* aff_thresh (``float``): Affinity threshold.

**returns**:

* True if the detector is valid, False otherwise.

---

### Function bnsa_class_prediction(...)

```python
def bnsa_class_prediction(
    features: npt.NDArray,
    class_detectors: npt.NDArray,
    aff_thresh: float
) -> int:
```

Defines the class of a sample from the non-self detectors.

**Parameters**:

* features (``npt.NDArray``): binary sample to be classified (shape: [n_features]).
* class_detectors (``npt.NDArray``): Array containing the detectors of all classes
(shape: [n_classes, n_detectors, n_features]).
* aff_thresh (``float``): Affinity threshold that determines whether a detector recognizes the sample as non-self.

**returns**:

* int: Index of the predicted class. Returns -1 if it is non-self for all classes.

---

### Function check_detector_rnsa_validity(...)

```python
def check_detector_rnsa_validity(
    x_class: npt.NDArray,
    vector_x: npt.NDArray,
    threshold: float,
    metric: int,
    p: float
) -> bool:
```

Checks the validity of a candidate detector (vector_x) against samples from a class (x_class) using the Hamming distance. A detector is considered INVALID if its distance to any sample in ``x_class`` is less than or equal to ``aff_thresh``.

**Parameters**:

* x_class (``npt.NDArray``): Array containing the class samples. Expected shape:  (n_samples, n_features).
* vector_x (``npt.NDArray``): Array representing the detector. Expected shape: (n_features,).
* threshold (``float``): threshold.
* metric (``int``): Distance metric to be used. Available options: [0 (Euclidean), 1 (Manhattan), 2 (Minkowski)]
* p (``float``): Parameter for the Minkowski distance (used only if `metric` is "minkowski").

**returns**:

* int: Index of the predicted class. Returns -1 if it is non-self for all classes.

---
