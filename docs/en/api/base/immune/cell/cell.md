---
id: cell
sidebar_label: Cell
keywords:
    - vector representation
    - cell
    - immune
    - immune cell
    - base class
    - dataclass
---

# Cell

Represents a basic immune cell.

> **Module:** `aisp.base.immune.cell`  
> **Import:** `from aisp.base.immune.cell import Cell`

---

## Attributes

| Name     | Type         | Default | Description                |
|----------|--------------|:-------:|----------------------------|
| `vector` | `np.ndarray` |    -    | A vector of cell features. |

---

## Methods

* `__eq__(other)`: Check if two cells are equal based on their vectors.
* `__array__()`: Array interface to NumPy, allows the instance to be treated as a `np.ndarray`.
* `__getitem__(item)`: Get elements from the feature vector using indexing.

