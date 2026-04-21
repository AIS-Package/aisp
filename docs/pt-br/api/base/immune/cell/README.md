---
id: immune-cell
sidebar_label: cell
keywords:
    - vector representation
    - cell
    - immune
    - immune cell
    - base class
    - bcell
    - antibody
    - dataclass
---

# aisp.base.immune.cell

Representação de células do sistema imunológico.

> **Módulos:** `aisp.base.immune.cell`

## Visão geral

Este módulo define as representações de células dos sistemas imunológicos artificiais e as implementa como dataclass.

## Classes

| Class                       | Descrição                                          |
|-----------------------------|----------------------------------------------------|
| [`Cell`](./cell.md)         | Representa uma célula imune básica.                |
| [`BCell`](./b-cell.md)      | Representa uma célula-B de memória.                |
| [`Antibody`](./antibody.md) | Representa um anticorpo.                           |
| [`Detector`](./detector.md) | Representa um detector não-próprio da classe rnsa. |