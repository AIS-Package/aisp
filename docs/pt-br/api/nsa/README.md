---
id: nsa
sidebar_label: aisp.nsa
keywords:
  - immune
  - sistemas imunológicos artificiais
  - classificação
  - seleção negativa
  - características binárias
  - detecção de anomalias
  - reconhecimento de não-próprio
  - reconhecimento de padrões
  - multiclasse
  - valores reais
  - v-detector
---

# aisp.nsa

Módulo com Algoritmo de Seleção Negativa (NSA).

> **Módulo:** `aisp.nsa`

## Visão geral

NSAs simulam o processo de maturação das células T no sistema imunológico, no qual essas células aprendem a distinguir
entre próprio e não-próprio. Apenas as células T capazes de reconhecer elementos não-próprios são preservadas.

## Classes

| Class               | Descrição                                                                                    |
|---------------------|----------------------------------------------------------------------------------------------|
| [`RNSA`](./rnsa.md) | Um algoritmo de aprendizado supervisionado para classificação de dados com valores reais.    |
| [`BNSA`](./bnsa.md) | Um algoritmo de aprendizado supervisionado para classificação de dados com valores binárias. |
