# AiNet (Artificial Immune Network)

This class extends the [**Base**](../../advanced-guides/base/clusterer.md) class.

## AiNet Constructor

The ``AiNet`` class implements the Artificial Immune Network algorithm for **compression** and **clustering**.
It uses principles from immune network theory, clonal selection, and affinity maturation to compress datasets and find clusters.

For clustering, it optionally uses a **Minimum Spanning Tree (MST)** to separate distant nodes into groups.

**Attributes:**

* **N** (``int``): Number of memory cells (antibodies) in the population. Defaults to 50.
* **n_clone** (``int``): Number of clones generated per selected memory cell. Defaults to 10.
* **top_clonal_memory_size** (``Optional[int]``): Number of highest-affinity antibodies selected for cloning. Defaults to 5.
* **n_diversity_injection** (``int``): Number of new random antibodies injected to maintain diversity. Defaults to 5.
* **affinity_threshold** (``float``): Threshold for cell selection/suppression. Defaults to 0.5.
* **suppression_threshold** (``float``): Threshold for removing similar memory cells. Defaults to 0.5.
* **mst_inconsistency_factor** (``float``): Factor to determine inconsistent MST edges. Defaults to 2.0.
* **max_iterations** (``int``): Maximum number of training iterations. Defaults to 10.
* **k** (``int``): Number of nearest neighbors used for label prediction. Defaults to 3.
* **metric** (Literal["manhattan", "minkowski", "euclidean"]): Way to calculate the distance between the detector and the sample:
  * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression:
    √( (x₁ - x₂)² + (y₁ - y₂)² + ... + (yn - yn)²).
  * ``'minkowski'`` ➜ The calculation of the distance is given by the expression:
    ( |X₁ - Y₁|p + |X₂ - Y₂|p + ... + |Xn - Yn|p) ¹/ₚ.
  * ``'manhattan'`` ➜ The calculation of the distance is given by the expression:
    ( |x₁ - x₂| + |y₁ - y₂| + ... + |yn - yn|).
    Defaults to "Euclidean".

* **seed** (``Optional[int]``): Seed for random number generation. Defaults to None.
* **use_mst_clustering** (``bool``): Whether to perform MST-based clustering. Defaults to True.
* **kwargs**:
  * **p** (``float``): Parameter for Minkowski distance. Defaults to 2.

**Other initialized variables:**

* **_population_antibodies** (``npt.NDArray``): Stores the current set of antibodies.
* **_memory_network** (``dict``): Dictionary mapping clusters to antibodies.
* **_mst_structure** (``scipy.sparse.csr_matrix``): MST adjacency structure.
* **_mst_mean_distance** (``float``): Mean of MST edge distances.
* **_mst_std_distance** (``float``): Standard deviation of MST edge distances.
* **classes** (``list``): List of cluster labels.

---

## Public Methods

### Function fit(...)

Trains the AiNet model on input data:

```python
def fit(self, X: npt.NDArray, verbose: bool = True):
````

**Input parameters:**

* **X**: Array with input samples (rows) and features (columns).
* **verbose**: Boolean, default True, enables progress feedback.

*Returns the class instance.*

---

### Function predict(...)

Predicts cluster labels for new samples:

```python
def predict(self, X) -> Optional[npt.NDArray]:
```

**Input parameters:**

* **X**: Array of input features.

**Returns:**

* **Predictions**: Array of cluster labels, or None if clustering is disabled.

---

### Function update_clusters(...)

Partitions clusters using the MST:

```python
def update_clusters(self, mst_inconsistency_factor: Optional[float] = None):
```

**Input parameters:**

* **mst_inconsistency_factor**: Optional float to override the MST inconsistency factor.

**Updates:**

* **_memory_network**: Dictionary of cluster labels to antibody arrays.
* **classes**: List of cluster labels.

---

## Private Methods

### Function _init_population_antibodies(...)

Initializes antibody population randomly.

```python
def _init_population_antibodies(self) -> npt.NDArray:
````

**Input parameters:** None

**Returns:** Initialized antibodies (`npt.NDArray`).

---

### Function _select_and_clone_population(...)

Selects top antibodies and generates mutated clones:

```python
def _select_and_clone_population(self, antigen: npt.NDArray, population: npt.NDArray) -> list:
```

**Input parameters:**

* **antigen**: Array representing the antigen to which affinities will be computed.
* **population**: Array of antibodies to be evaluated and cloned.

**Returns:** List of mutated clones.

---

### Function _clonal_suppression(...)

Suppresses redundant clones based on thresholds:

```python
def _clonal_suppression(self, antigen: npt.NDArray, clones: list):
```

**Input parameters:**

* **antigen**: Array representing the antigen.
* **clones**: List of candidate clones to be suppressed.

**Returns:** List of non-redundant, high-affinity clones.

---

### Function _memory_suppression(...)

Removes redundant antibodies from memory pool:

```python
def _memory_suppression(self, pool_memory: list) -> list:
```

**Input parameters:**

* **pool_memory**: List of antibodies currently in memory.

**Returns:** Cleaned memory pool (`list`).

---

### Function _diversity_introduction(...)

Introduces new random antibodies:

```python
def _diversity_introduction(self) -> npt.NDArray:
```

**Input parameters:** None

**Returns:** Array of new antibodies (`npt.NDArray`).

---

### Function _affinity(...)

Calculates stimulus between two vectors:

```python
def _affinity(self, u: npt.NDArray, v: npt.NDArray) -> float:
```

**Input parameters:**

* **u**: Array representing the first point.
* **v**: Array representing the second point.

**Returns:** Affinity score (`float`) in [0,1].

---

### Function _calculate_affinities(...)

Calculates affinity matrix between reference and target vectors:

```python
def _calculate_affinities(self, u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
```

**Input parameters:**

* **u**: Reference vector (`npt.NDArray`) of shape `(n_features,)`.
* **v**: Target vectors (`npt.NDArray`) of shape `(n_samples, n_features)`.

**Returns:** Array of affinities (`npt.NDArray`) with shape `(n_samples,)`.

---

### Function _clone_and_mutate(...)

Generates mutated clones:

```python
def _clone_and_mutate(self, antibody: npt.NDArray, n_clone: int) -> npt.NDArray:
```

**Input parameters:**

* **antibody**: Original antibody vector to clone and mutate.
* **n_clone**: Number of clones to generate.

**Returns:** Array of mutated clones (`npt.NDArray`) of shape `(n_clone, len(antibody))`.

---

### Function _build_mst(...)

Constructs the MST and stores statistics.

```python
def _build_mst(self):
```

**Input parameters:** None

**Raises:** ValueError if antibody population is empty.

**Updates internal variables:**

* **_mst_structure**: MST adjacency structure.
* **_mst_mean_distance**: Mean edge distance.
* **_mst_std_distance**: Standard deviation of MST edge distances.

---

## References

> 1. De Castro, Leandro & José, Fernando & von Zuben, Antonio Augusto. (2001). aiNet: An Artificial Immune Network for Data Analysis.
>    Available at: [https://www.researchgate.net/publication/228378350_aiNet_An_Artificial_Immune_Network_for_Data_Analysis](https://www.researchgate.net/publication/228378350_aiNet_An_Artificial_Immune_Network_for_Data_Analysis)

> 2. SciPy Documentation. *Minimum Spanning Tree*.
>    Available at: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree)
