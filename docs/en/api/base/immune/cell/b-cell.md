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
| `vector` | `np.ndarray` |    -    | A vector of cell features. |

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

Generate **N** clones of the current cell and apply hypermutation to the clones.

**Parameters**

| Name           | Type                                                 |         Default         | Description                                            |
|----------------|------------------------------------------------------|:-----------------------:|--------------------------------------------------------|
| `n`            | `int`                                                |            -            | Number of clones to generate from the original b-cell. |
| `feature_type` | [`FeatureType`](../../../utils/types.md#featuretype) | `"continuous-features"` | Specifies the type of features of the cell.            |
| `bounds`       | `Optional[npt.NDArray[np.float64]]`                  |         `None`          | Array (n_features, 2) with min and max per dimension.  |

**Returns**

| Type          | Description                                                   |
|---------------|---------------------------------------------------------------|
| `npt.NDArray` | An array containing N mutated vectors from the original cell. |
