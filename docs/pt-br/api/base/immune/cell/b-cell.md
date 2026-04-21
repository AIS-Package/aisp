---
id: b-cell
sidebar_label: BCell
keywords:
  - célula-b
  - memória imune
  - dataclass
  - mutação clonal
  - expansão clonal
---

# BCell

Representa uma célula-B de memória.

:::tip[Herança]

Esta classe herda de [Cell](./cell.md)

:::

> **Módulo:** `aisp.base.immune.cell`  
> **Importação:** `from aisp.base.immune.cell import BCell`

---

## Atributos

| Nome     | Tipo         | Padrão | Descrição                               |
|----------|--------------|:------:|-----------------------------------------|
| `vector` | `np.ndarray` |   -    | Vetor com as características da célula. |

---

## Métodos Públicos

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

Gera **N** clones da célula atual e aplica hipermutação aos clones.

**Parâmetros**

| Nome           | Tipo                                                 |         Padrão          | Descrição                                                       |
|----------------|------------------------------------------------------|:-----------------------:|-----------------------------------------------------------------|
| `n`            | `int`                                                |            -            | Numero de clones que serão gerados a partir da célula original. |
| `feature_type` | [`FeatureType`](../../../utils/types.md#featuretype) | `"continuous-features"` | Especifica o tipo de características da célula.                 |
| `bounds`       | `Optional[npt.NDArray[np.float64]]`                  |         `None`          | Matriz (n_features, 2) com o min e o max de cada dimensão.      |

**Returns**

| Tipo          | Descrição                                                |
|---------------|----------------------------------------------------------|
| `npt.NDArray` | Uma matriz contendo N clones mutados da célula original. |
