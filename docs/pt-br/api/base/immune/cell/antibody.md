---
id: antibody
sidebar_label: Antibody
keywords:
    - anticorpo
    - afinidade
    - célula
    - imune
    - dataclass
---

# Antibody

Representa um anticorpo.

:::tip[Herança]

Esta classe herda de [Cell](./cell.md)

:::

> **Módulo:** `aisp.base.immune.cell`  
> **Importação:** `from aisp.base.immune.cell import Antibody`

---

## Atributos

| Nome       | Tipo          | Padrão | Descrição                              |
|------------|---------------|:------:|----------------------------------------|
| `vector`   | `npt.NDArray` |   -    | Vetor com as características anticorpo |
| `affinity` | `float`       |   -    | Valor da afinidade do anticorpo        |

---

## Métodos

* `__lt__(other)`: Compara a célula atual com outra célula `Antibody` com base na afinidade.
* `__eq__(other)`: Verifica se o anticorpo possui a mesma afinidade do outro.
