---
id: populations
sidebar_label: populations
keywords: 
    - binary
    - classifying
    - affinity threshold
    - real-Valued
    - memory B-cell
    - clonal Expansion
    - populations
---

# populations

Provide utility functions for generating antibody populations in immunological algorithms.

> **Module:** `aisp.base.immune`  
> **Import:** `from aisp.base.immune import populations`

## Methods

### generate_random_antibodies

````python
def generate_random_antibodies(
    n_samples: int,
    n_features: int,
    feature_type: FeatureTypeAll = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray:
    ...
````

Generate a random antibody population.

**Parameters**

| Name           | Type                      |         Default         | Description                                                                                                             |
|----------------|---------------------------|:-----------------------:|-------------------------------------------------------------------------------------------------------------------------|
| `n_samples`    | `int`                     |            -            | Number of antibodies (samples) to generate.                                                                             |
| `n_features`   | `int`                     |            -            | Number of features (dimensions) for each antibody.                                                                      |
| `feature_type` | `FeatureType`             | `"continuous-features"` | Specifies the type of features: "continuous-features", "binary-features", "ranged-features", or "permutation-features". |
| `bounds`       | `npt.NDArray[np.float64]` |         `None`          | Array (n_features, 2) with min and max per dimension.                                                                   |

**Raises**

`ValueError` - If number of features must be greater than zero.

**Returns**

`npt.NDArray` - Array of shape (n_samples, n_features) containing the generated antibodies.
