# Cell Classes

Representation of immune system cells.

## Cell

Represents a basic immune cell.

```python
@dataclass(slots=True)
class Cell:
    vector: np.ndarray
```

### Attributes
* **vector** (`np.ndarray`): A vector of cell features.

### Methods
* `__eq__(other)`: Check if two cells are equal based on their vectors.
* `__array__()`: Array interface to NumPy, allows the instance to be treated as a `np.ndarray`.
* `__getitem__(item)`: Get elements from the feature vector using indexing.

---

## BCell

Represents a memory B-cell.

```python
@dataclass(slots=True, eq=False)
class BCell(Cell):
    vector: np.ndarray
```

### Methods

### hyper_clonal_mutate(...)

```python
def hyper_clonal_mutate(
    self,
    n: int,
    feature_type: FeatureType = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None
) -> np.ndarray
```

Clones N features from a cell's features, generating a set of mutated vectors.

#### Parameters
* **n** (`int`): Number of clones to be generated from mutations of the original cell.
* **feature_type** (`Literal["binary-features", "continuous-features", "ranged-features"]`): 
  Specifies the type of feature_type to use based on the nature of the input features
* **bounds** (`Optional[npt.NDArray[np.float64]]`): Array (n_features, 2) with min and max per dimension.

#### Returns
* **npt.NDArray**: An array containing N mutated vectors from the original cell.

---

## Antibody

Represent an antibody.

```python
@dataclass(slots=True)
class Antibody(Cell):
    vector: np.ndarray
    affinity: float
```

### Attributes
* **vector** (`npt.NDArray`): A vector of cell features.
* **affinity** (`float`): Affinity value for the antibody.

### Methods
* `__lt__(other)`: Compare this cell with another Antibody cell based on affinity.
* `__eq__(other)`: Check if this cell has the same affinity as another cell.

---

## Detector

Represents a non-self detector of the RNSA class.

```python
@dataclass(slots=True)
class Detector:
    position: npt.NDArray[np.float64]
    radius: Optional[float] = None
```

### Attributes
* **position** (`npt.NDArray[np.float64]`): Detector feature vector.
* **radius** (`Optional[float]`): Detector radius, used in the V-detector algorithm.
