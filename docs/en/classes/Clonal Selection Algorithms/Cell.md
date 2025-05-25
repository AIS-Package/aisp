## Cell Class:

Represents a memory B-cell.

### Constructor:

Parameters:
* vector (``Optional[npt.NDArray]``): A feature vector of the cell. Defaults to None.

---

### Function hyper_clonal_mutate(...):

Parameters:
* n (``int``): The number of clones to be generated from mutations in the original cell.
* algorithm (``Literal["continuous-features", "binary-features"]``): Specifies the type of
algorithm to use based on the nature of the input features

```python
def hyper_clonal_mutate(
    self,
    n: int,
    algorithm: Literal[
      "continuous-features",
      "binary-features"
    ] = "continuous-features"
) -> npt.NDArray
```

Returns an array containing N mutated vectors from the original cell.