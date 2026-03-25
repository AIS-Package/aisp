---
id: base-classifier
sidebar_label: BaseClassifier
keywords:
  - base
  - classificação
  - classificação interface
  - pontuação de acurácia
  - fit
  - predict
tags:
  - classificador
  - classificação
---

# BaseClassifier

Classe base abstrata para algoritmos de classificação.

> **Módulo:** `aisp.base`  
> **Importação:** `from aisp.base import BaseClassifier`

---

## Visão geral

Esta classe define a interface principal para algoritmos de classificação.  
Ela define a implementação dos metodos `fit` e `predict` em todas as classes derivadas, e fornece uma implementação
do método `score`.

Caso de uso:

- Classe base abstrata para estender classes de algoritmos de classificação.

---

## Atributos

| Nome      | Tipo                    | Padrão | Descrição                                                      |
|-----------|-------------------------|:------:|----------------------------------------------------------------|
| `classes` | `Optional[npt.NDArray]` | `None` | Rótulos das classes identificado de `y` durante o treinamento. |

---

## Métodos abstratos

### fit

```python
@abstractmethod
def fit(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list],
    verbose: bool = True
) -> BaseClassifier:
    ...
```

Treine o modelo usando os dados de entrada X e seus rótulos correspondentes y.
Este método abstrato é implementado é responsabilidade das classes filhas.

**Parâmetros**

| Nome      | Tipo                       | Padrão | Descrição                                                            |
|-----------|----------------------------|:------:|----------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |   -    | Dados de entrada utilizados para o treinamento.                      |
| `y`       | `Union[npt.NDArray, list]` |   -    | Rótulos correspondentes as características dos dados de entrada.     |
| `verbose` | `bool`                     | `True` | Indica se as mensagens de progresso do treinamento deve ser exibido. |

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

Gera previsões com base nos dados de entrada X.  
Este método abstrato é implementado é responsabilidade das classes filhas.

**Parâmetros**

| Nome | Tipo                       | Padrão | Descrição                                         |
|------|----------------------------|:------:|---------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |   -    | Dados de entrada que serão previstos pelo modelo. |

**Returns**

| Tipo          | Descrição                                        |
|---------------|--------------------------------------------------|
| `npt.NDArray` | Array com as previsões para cada amostra de `X`. |

---

## Métodos públicos

### score

```python
def score(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list]
) -> float:
    ...
```

A função calcula o desempenho do modelo nas previsões utilizando a métrica de acurácia.

Esta função realiza a previsão de X e verifica quantos elementos são iguais entre o vetor y e y_predicted.  
Esta função foi adicionada para compatibilidade com algumas funções do scikit-learn.

**Parâmetros**

| Nome | Tipo                       | Padrão | Descrição                                                          |
|------|----------------------------|:------:|--------------------------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |   -    | Conjunto de características com dimensões (n_samples, n_features). |
| `y`  | `Union[npt.NDArray, list]` |   -    | Rótulos verdadeiros com dimensão (n_amostras,).                    |

**Returns**

| Tipo    | Descrição             |
|---------|-----------------------|
| `float` | A precisão do modelo. |
