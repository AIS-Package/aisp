---
id: base-clusterer
sidebar_label: BaseClusterer
keywords:
  - base
  - clusterer
  - clusterer interface
  - cluster labels
  - fit
  - predict
  - fit_predict
  - agrupamento
tags:
  - clusterer
  - agrupamento
---

# BaseClusterer

Classe base abstrata para algoritmos de clustering.

> **Módulos:** `aisp.base`  
> **Importação:** `from aisp.base import BaseClusterer`

---

## Visão geral

Esta classe define a interface principal para algoritmos de clusterização.  
Ela define a implementação dos metodos fit e predict em todas as classes filhas, e fornece a implementação do
método `fit_predict`.

Casos de uso:

- Classe base abstrata para estender classes de algoritmos de clusterização.

---

## Atributos

| Nome     | Tipo                    | Padrão | Descrição                                               |
|----------|-------------------------|:------:|---------------------------------------------------------|
| `labels` | `Optional[npt.NDArray]` | `None` | Rótulos dos clusters encontrados durante o treinamento. |

---

## Métodos abstratos

### fit

```python
@abstractmethod
def fit(self, X: Union[npt.NDArray, list], verbose: bool = True) -> BaseClusterer:
    ...
```

Treinamento do modelo utilizando os dados de entrada `X`.  
Este método abstrato é implementado é responsabilidade das classes filhas.

**Parâmetros**

| Nome      | Tipo                       | Padrão | Descrição                                                                                                      |
|-----------|----------------------------|:------:|----------------------------------------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |   -    | Amostras de entrada para treinamento. Cada linha corresponde a uma amostra e cada coluna a uma característica. |
| `verbose` | `bool`                     | `True` | Se True, exibe informações sobre o progresso do treinamento.                                                   |

**Returns**

| Tipo   | Descrição                      |
|--------|--------------------------------|
| `Self` | Retorna a instancia da classe. |

---

### predict

```python
@abstractmethod
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
    ...
```

Gera previsões com base nos dados de entrada `X`.  
Este método abstrato é implementado é responsabilidade das classes filhas.

**Parâmetros**

| Nome | Tipo                       | Padrão | Descrição                                                                              |
|------|----------------------------|:------:|----------------------------------------------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |   -    | Amostras de entrada. Deve ter o mesmo número de características usadas no treinamento. |

**Returns**

| Tipo          | Descrição                                                   |
|---------------|-------------------------------------------------------------|
| `npt.NDArray` | Os labels do cluster previsto para cada amostra de entrada. |

---

## Métodos públicos

### fit_predict

```python
def fit_predict(self, X: Union[npt.NDArray, list], verbose: bool = True) -> npt.NDArray:
    ...
```

Ajusta o modelo e com os dados de X e retorna os labels para cada amostra de X.  
Este é um método que combina `fit` e `predict` em uma única chamada.

**Parâmetros**

| Nome      | Tipo                       | Padrão | Descrição                                                                                                      |
|-----------|----------------------------|:------:|----------------------------------------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |   -    | Amostras de entrada para treinamento. Cada linha corresponde a uma amostra e cada coluna a uma característica. |
| `verbose` | `bool`                     | `True` | Se True, exibe informações sobre o progresso do treinamento.                                                   |

**Returns**

| Tipo          | Descrição                                                   |
|---------------|-------------------------------------------------------------|
| `npt.NDArray` | Os labels do cluster previsto para cada amostra de entrada. |
