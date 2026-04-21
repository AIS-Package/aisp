---
id: airs
sidebar_label: AIRS
keywords:
  - classificação
  - sistema imunológico artificial de reconhecimento
  - células de memória
  - k-nn
  - aprendizado supervisionado
  - AIRS2
  - seleção clonal
tags:
  - classificação
  - supervisionado
  - seleção clonal
---

# AIRS

Sistema Imunológico Artificial de Reconhecimento (AIRS)

:::tip[Herança]

Esta classe herda de [BaseClassifier](../base/base-classifier.md)

:::


> **Módulo:** `aisp.csa`  
> **Importação:** `from aisp.csa import AIRS`

---

## Visão geral

O _Artificial Immune Recognition System_ (AIRS) é um algoritmo de classificação inspirado no processo de seleção
clonal. Esta implementação é baseada na versão simplificada (AIRS2) descrita em [^1]. O algoritmo foi adaptado
para suportar amostras com características com valores reais (contínuos) e binários (discretos).

:::note

Esta implementação é inspirada no AIRS2, uma versão simplificada do algoritmo AIRS original, introduzindo
adaptações para lidar com conjuntos de dados contínuos e binários.

Baseado no Algorithm 16.5 de Brabazon et al. [^1]

Trabalhos relacionados e notáveis: [^2].

:::

---

## Exemplo

```python
import numpy as np
from aisp.csa import AIRS

np.random.seed(1)
# Gerando dados para o treinamento
a = np.random.uniform(high=0.5, size=(50, 2))
b = np.random.uniform(low=0.51, size=(50, 2))
x_train = np.vstack((a, b))
y_train = [0] * 50 + [1] * 50
# Instancia do AIRS
airs = AIRS(n_resources=5, rate_clonal=5, rate_hypermutation=0.65, seed=1)
airs = airs.fit(x_train, y_train, verbose=False)
x_test = [
    [0.15, 0.45],  # Expected: Class 0
    [0.85, 0.65],  # Esperado: Classe 1
]
y_pred = airs.predict(x_test)
print(y_pred)
```

Output:

```bash
[0 1]
```

---

## Parâmetros do Construtor

| Nome                        | Tipo    |    Default    | Descrição                                                                                                                                                      |
|-----------------------------|---------|:-------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `n_resources`               | `float` |     `10`      | Quantidade total de recursos disponíveis.                                                                                                                      |
| `rate_clonal`               | `int`   |     `10`      | Número máximo de clones possíveis de uma classe. Esta quantidade é multiplicada por (estímulo da célula * rate_hypermutation) para definir o número de clones. |
| `rate_mc_init`              | `float` |     `0.2`     | Porcentagem de amostras usadas para inicializar a população de células de memória.                                                                             |
| `rate_hypermutation`        | `float` |    `0.75`     | Taxa de clones mutados derivada de rate_clonal como um fator escalar.                                                                                          |
| `affinity_threshold_scalar` | `float` |    `0.75`     | Limiar de afinidade normalizado.                                                                                                                               |
| `k`                         | `int`   |      `3`      | Número de vizinhos mais próximos (k-NN) que será usado para escolher um rótulo na predição.                                                                    |
| `max_iters`                 | `int`   |     `100`     | Número máximo de interações no processo de refinamento do conjunto ARB exposto a aᵢ.                                                                           |
| `resource_amplified`        | `float` |     `1.0`     | Amplificador de consumo de recursos, multiplicado com o estímulo para subtrair recursos.                                                                       |
| `metric`                    | `str`   | `"euclidean"` | Métrica de distância usada para calcular a afinidade entre células e amostras.                                                                                 |
| `seed`                      | `int`   |    `None`     | Seed para geração aleatória.                                                                                                                                   |
| `p`                         | `float` |      `2`      | Este parâmetro é usado na distância de Minkowski.                                                                                                              |

## Atributos

| Nome           | Tipo                                      | Padrão | Descrição                                  |
|----------------|-------------------------------------------|:------:|--------------------------------------------|
| `cells_memory` | `Optional[Dict[str \| int, list[BCell]]]` |   -    | Armazena as células de memória por classe. |

---

## Métodos Públicos

### fit

```python
def fit(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list],
    verbose: bool = True,
) -> AIRS:
    ...
```

Treina o modelo com os dados de entrada utilizando o algoritmo AIRS2.

A função `fit(...)`, realiza o treinamento de acordo com X e y, usando o método AIRS.

**Parâmetros**

| Nome      | Tipo                       | Padrão | Descrição                                                                                                      |
|-----------|----------------------------|:------:|----------------------------------------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |   -    | Amostras de entrada para treinamento. Cada linha corresponde a uma amostra e cada coluna a uma característica. |
| `y`       | `Union[npt.NDArray, list]` |   -    | Vetor alvo no formato (n_samples,). Deve conter o mesmo número de amostras que X.                              |
| `verbose` | `bool`                     | `True` | Se True, exibe informações sobre o progresso do treinamento.                                                   |

**Returns**

| Tipo   | Descrição                      |
|--------|--------------------------------|
| `Self` | Retorna a instancia da classe. |

**Exceções**

| Exceção     | Descrição                                                     |
|-------------|---------------------------------------------------------------|
| `TypeError` | Se X ou y não forem arrays ou tiverem tamanhos incompatíveis. |

---

### predict

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
    ...
```

Prevê os rótulos dos dados de testes com base nas células de memórias criadas durante o treinamento.

Este método utiliza as células de memórias para classificar os dados de entrada usando a abordagem dos k-vizinhos mais
próximos (K-NN).

**Parâmetros**

| Nome | Tipo                       | Padrão | Descrição                                                                              |
|------|----------------------------|:------:|----------------------------------------------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |   -    | Amostras de entrada. Deve ter o mesmo número de características usadas no treinamento. |

**Returns**

| Tipo          | Descrição                                                            |
|---------------|----------------------------------------------------------------------|
| `npt.NDArray` | Array no formato `n_samples` contendo as classes previstas para `X`. |

**Exceções**

| Exceção                                                                 | Descrição                                                                                 |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `TypeError`                                                             | Se X não for um ndarray ou list.                                                          |
| [`FeatureDimensionMismatch`](../exceptions.md#featuredimensionmismatch) | Se o número de características em X não corresponder ao esperado.                         |
| [`ModelNotFittedError`](../exceptions.md#modelnotfittederror)           | Se o modelo ainda não tiver sido treinado e não possuir o conjunto de células de memoria. |

---

## Exemplos Estendidos

Exemplos completos de uso estão disponíveis nos notebooks Jupyter:

- [**Iris Dataset Example**](../../../../examples/en/classification/AIRS/iris_dataBase_example_en.ipynb)
- [**Geyser Dataset Example**](../../../../examples/en/classification/AIRS/geyser_dataBase_example_en.ipynb)
- [**Mushrooms Dataset Example**](../../../../examples/en/classification/AIRS/mushrooms_dataBase_example_en.ipynb)
- [**Random Dataset Example**](../../../../examples/en/classification/AIRS/example_with_randomly_generated_dataset-en.ipynb)

---

## Referências

[^1]: Brabazon, A., O'Neill, M., & McGarraghy, S. (2015). Natural Computing Algorithms. In Natural Computing Series.
    Springer Berlin Heidelberg. [https://doi.org/10.1007/978-3-662-43631-8](https://doi.org/10.1007/978-3-662-43631-8)

[^2]: AZZOUG, Aghiles. Artificial Immune Recognition System V2. Available at:
    [https://github.com/AghilesAzzoug/Artificial-Immune-System](https://github.com/AghilesAzzoug/Artificial-Immune-System)
