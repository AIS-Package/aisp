---
id: detector
sidebar_label: Detector
keywords: 
    - detector
    - cell
    - immune
    - radius
    - non-self
    - nsa
    - dataclass
---

# Detector

Represents a non-self detector of the RNSA class.

> **Module:** `aisp.base.immune.cell`  
> **Import:** `from aisp.base.immune.cell import Detector`

---

## Attributes

| Name       | Type                      | Default | Description                                        |
|------------|---------------------------|:-------:|----------------------------------------------------|
| `position` | `npt.NDArray[np.float64]` |    -    | Detector feature vector.                           |
| `radius`   | `float, optional`         |    -    | Detector radius, used in the V-detector algorithm. |
