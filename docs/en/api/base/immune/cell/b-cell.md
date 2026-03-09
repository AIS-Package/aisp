---
id: b-cell
sidebar_label: BCell
keywords:
    - B-cell
    - immune memory
    - dataclass
    - clonal mutation
    - clonal expansion
---

# BCell

Represents a memory B-cell.

:::tip[Inheritance]

This class extends [Cell](./cell.md)

:::

> **Module:** `aisp.base.immune.cell`  
> **Import:** `from aisp.base.immune.cell import BCell`

---
## Attributes

| Name     | Type         | Default | Description                |
|----------|--------------|:-------:|----------------------------|
| `vector` | `np.ndarray` |    -    | A vector of cell features. |]

---

## Public Methods

### hyper_clonal_mutate

```python
def hyper_clonal_mutate(
    self,
    n: int,
    feature_type: FeatureType = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray:
    ...
```

Clones N features from a cell's features, generating a set of mutated vectors.

**Parameters**

| Name           | Type                                     |         Default         | Description                                                                         |
|----------------|------------------------------------------|:-----------------------:|-------------------------------------------------------------------------------------|
| `n`            | `int`                                    |            -            | Number of clones to be generated from mutations of the original cell.               |
| `feature_type` | [`FeatureType`](../../../utils/types.md) | `"continuous-features"` | Specifies the type of feature_type to use based on the nature of the input features |
| `bounds`       | `Optional[npt.NDArray[np.float64]]`      |         `None`          | Array (n_features, 2) with min and max per dimension.                               |

**Returns**

`npt.NDArray` - An array containing N mutated vectors from the original cell.
