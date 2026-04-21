---
id: csa
sidebar_label: aisp.csa
keywords: 
    - sistema imunológico
    - seleção clonal
    - clonalg
    - airsv2
    - proliferação de anticorpos
    - mutações
    - algoritmos de seleção clonal
    - sistemas imunológicos artificiais
    - classificação
    - otimização
---

# aisp.csa

Módulo com algoritmos de seleção clonal (ASC).

> **Módulo:** `aisp.csa`

## Visão geral

Os **ASCs** são inspirados no processo de proliferação de anticorpos ao detectar o antígeno, no qual os clones gerados
passam por mutações na tentativa de melhorar o reconhecimento dos patógenos.

## Classes

| Class                     | Descrição                                                                                                                                                     |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`AIRS`](./airs.md)       | Um algoritmos supervisionado para tarefas de classificação.                                                                                                   |
| [`Clonalg`](./clonalg.md) | Esta implementação do ASC para otimização, foi adaptado tanto para minimização quanto maximização de custos em problemas binários, contínuos e de permutação. |
