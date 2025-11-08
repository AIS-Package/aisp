# Populations Module

Utility functions for generating antibody populations in immunological algorithms.

## generate_random_antibodies(...)

```python
def generate_random_antibodies(
    n_samples: int,
    n_features: int,
    feature_type: FeatureTypeAll = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray
```

Generate a random antibody population.

### Parameters

* **n_samples** (`int`): Number of antibodies (samples) to generate.
* **n_features** (`int`): Number of features (dimensions) for each antibody.
* **feature_type** (`FeatureTypeAll`, default="continuous-features"): Specifies the type of features: "continuous-features", "binary-features", "ranged-features", or "permutation-features".
* **bounds** (`Optional[npt.NDArray[np.float64]]`): Array (n_features, 2) with min and max per dimension.

### Returns

* **npt.NDArray**: Array of shape (n_samples, n_features) containing the generated antibodies.

### Raises

* **ValueError**: If the number of features is less than or equal to zero.