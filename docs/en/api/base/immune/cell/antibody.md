---
id: antibody
sidebar_label: Antibody
keywords:
    - antibody
    - affinity
    - cell
    - immune
    - dataclass
---

# Antibody

Represent an antibody.

:::tip[Inheritance]

This class extends [Cell](./cell.md)

:::

> **Module:** `aisp.base.immune.cell`  
> **Import:** `from aisp.base.immune.cell import Antibody`

---

## Attributes

| Name       | Type          | Default | Description                      |
|------------|---------------|:-------:|----------------------------------|
| `vector`   | `npt.NDArray` |    -    | A vector of cell features.       |
| `affinity` | `float`       |    -    | Affinity value for the antibody. |
