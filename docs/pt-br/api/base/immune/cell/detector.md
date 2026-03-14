---
id: detector
sidebar_label: Detector
keywords: 
    - detector
    - célula
    - imune
    - raio
    - não-próprio
    - nsa
    - dataclass
---

# Detector

Representa um detector não-próprio da classe rnsa.

> **Módulo:** `aisp.base.immune.cell`  
> **Importação:** `from aisp.base.immune.cell import Detector`

---

## Atributos

| Nome       | Tipo                      | Padrão | Descrição                                        |
|------------|---------------------------|:------:|--------------------------------------------------|
| `position` | `npt.NDArray[np.float64]` |   -    | Vetor com as características do detector.        |
| `radius`   | `float, optional`         |   -    | Raio do detector, usado no algoritmo V-detector. |
