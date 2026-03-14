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

Representa uma célula imune básica.

> **Módulo:** `aisp.base.immune.cell`  
> **Importação:** `from aisp.base.immune.cell import Cell`

---

## Atributos

| Nome     | Tipo         | Padrão | Descrição                               |
|----------|--------------|:------:|-----------------------------------------|
| `vector` | `np.ndarray` |   -    | Vetor com as características da célula. |

---

## Métodos

* `__eq__(other)`: Verifica se duas células são iguais com base nos seus vetores.
* `__array__()`: Interface de array Numpy, permite que a instância seja tratada como um `np.ndarray`
* `__getitem__(item)`: Obtém um elemento do vetor com base no index.

