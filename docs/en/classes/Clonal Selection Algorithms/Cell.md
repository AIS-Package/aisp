## Cell Class:

Represents a memory B-cell.

### Constructor:

Parameters:
* vector (``Optional[npt.NDArray]``): A feature vector of the cell. Defaults to None.

---

### Function hyper_clonal_mutate(...):

Parameters:
* n (``int``): The number of clones to be generated from mutations in the original cell.
* feature_type (``Literal["continuous-features", "binary-features", "ranged-features"]``): Specifies the type of
algorithm to use based on the nature of the input features
* bounds (``np.ndarray``): Array (n_features, 2) with min and max per dimension.

```python
def hyper_clonal_mutate(
    self,
    n: int,
    feature_type: Literal[
        "binary-features",
        "continuous-features",
        "ranged-features"
    ],
    bounds: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray
```

Returns an array containing N mutated vectors from the original cell.