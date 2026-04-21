---
id: ai-net
sidebar_label: AiNet
keywords:
    - rede imunológica
    - agrupamento
    - compressão de dados
    - aprendizado não supervisionado
    - Minimum Spanning Tree
tags:
    - agrupamento
    - não supervisionado
    - rede imunológica
---

# AiNet

Rede Imunológica Artificial (AiNet) para agrupamento e compressão.

:::tip[Herança]

Esta classe herda de [BaseClusterer](../base/base-clusterer.md).

:::


> **Módulo:** `aisp.ina`  
> **Importação:** `from aisp.ina import AiNet`

---

## Visão geral

Esta classe implementa o algoritmo AiNet é projetado para tarefas de agrupamento
e compressão de dados; O algoritmo é inspirado na teoria da rede imunológica, seleção clonal e maturação por afinidade para
reduzir conjuntos de dados com muitas amostras [^1].  
No agrupamento a classe utiliza a implementação da Scipy da **Árvore geradora mínima** para separa as arestas mais
distantes em grupos de dados [^2].

---

## Exemplo

```python
import numpy as np
from aisp.ina import AiNet

np.random.seed(1)
# Gerando dados de treinamento
a = np.random.uniform(high=0.4, size=(50, 2))
b = np.random.uniform(low=0.6, size=(50, 2))
x_train = np.vstack((a, b))
# Instância do AiNet
ai_net = AiNet(
    N=150,
    mst_inconsistency_factor=1,
    seed=1,
    affinity_threshold=0.85,
    suppression_threshold=0.7
)
ai_net = ai_net.fit(x_train, verbose=True)
x_test = [
    [0.15, 0.45],  # Esperado: rótulo 0
    [0.85, 0.65],  # Esperado: rótulo 1
]
y_pred = ai_net.predict(x_test)
print(y_pred)
```

**Output**

```bash
[0 1]
```

---

## Parâmetros do Construtor

| Nome                       | Tipo                                         |    Default    | Descrição                                                                                                                                                  |
|----------------------------|----------------------------------------------|:-------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `N`                        | `int`                                        |     `50`      | Número de células de memória (anticorpos) para iniciar a população.                                                                                        |
| `n_clone`                  | `int`                                        |     `10`      | Número de clones gerados para cada célula de memória selecionada.                                                                                          |
| `top_clonal_memory_size`   | `int`                                        |      `5`      | Número de anticorpos com maior afinidade que selecionados para a clonagem e mutação.                                                                       |
| `n_diversity_injection`    | `int`                                        |      `5`      | Número de novas células de memória aleatórias que serão inseridas para manter a diversidade.                                                               |
| `affinity_threshold`       | `float`                                      |     `0.5`     | Limiar de afinidade (similaridade) parta determinar a supressão das células.                                                                               |
| `suppression_threshold`    | `float`                                      |     `0.5`     | Limiar de supressão das células de memórias semelhantes.                                                                                                   |
| `mst_inconsistency_factor` | `float`                                      |     `2.0`     | Fator usado para determinar quais arestas da Árvore Geradora Mínima (MST) são consideradas inconsistentes.                                                 |
| `max_iterations`           | `int`                                        |     `10`      | Número máximo de iterações de treinamento.                                                                                                                 |
| `k`                        | `int`                                        |      `3`      | Número de vizinhos mais próximos usados para predição de rótulos.                                                                                          |
| `metric`                   | [`MetricType`](../utils/types.md#metrictype) | `"euclidean"` | Métrica de distância utilizada para calcular a similaridade entre as células de memória.                                                                   |
| `seed`                     | `Optional[int]`                              |    `None`     | Seed para geração aleatória.                                                                                                                               |
| `use_mst_clustering`       | `bool`                                       |    `True`     | Quando `True`, realiza o agrupamento utilizando a MST. Se `False`, o agrupamento não é executado e o método predict lança uma exceção ModelNotFittedError. |
| `p`                        | `float`                                      |     `2.0`     | Parâmetro `p` utilizado na distância de Minkowski.                                                                                                         |

## Atributos

| Nome                    | Tipo                    | Padrão | Descrição                                                                     |
|-------------------------|-------------------------|:------:|-------------------------------------------------------------------------------|
| `memory_network`        | `Dict[int, List[Cell]]` |   -    | Rede imunológica que representa os clusters.                                  |
| `population_antibodies` | `Optional[npt.NDArray]` |   -    | População de anticorpos de memória.                                           |
| `mst`                   | `dict`                  |   -    | Árvore geradora minima com estatísticas (graph, mean_distance, std_distance). |

---

## Métodos Públicos

### fit

```python
def fit(self, X: Union[npt.NDArray, list], verbose: bool = True) -> AiNet:
    ...
```

Treina o modelo com os dados de entrada utilizando o algoritmo AiNet.

**Parâmetros**

| Nome      | Tipo                       | Padrão | Descrição                                                                                                      |
|-----------|----------------------------|:------:|----------------------------------------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |   -    | Amostras de entrada para treinamento. Cada linha corresponde a uma amostra e cada coluna a uma característica. |
| `verbose` | `bool`                     | `True` | Se True, exibe informações sobre o progresso do treinamento.                                                   |

**Returns**

| Tipo   | Descrição                      |
|--------|--------------------------------|
| `Self` | Retorna a instancia da classe. |

**Exceções**

| Exceção                                                         | Descrição                                                |
|-----------------------------------------------------------------|----------------------------------------------------------|
| `TypeError`                                                     | Se `X` não for um ndarray ou list.                       |
| [`UnsupportedTypeError`](../exceptions.md#unsupportedtypeerror) | Se o tipo das características em X não forem suportados. |

---

### predict

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
    ...
```

Prever os rótulos dos clusters para os dados de entrada.

**Parâmetros**

| Nome | Tipo                       | Padrão | Descrição                                                                              |
|------|----------------------------|:------:|----------------------------------------------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |   -    | Amostras de entrada. Deve ter o mesmo número de características usadas no treinamento. |

**Returns**

| Tipo          | Descrição                                   |
|---------------|---------------------------------------------|
| `npt.NDArray` | Rótulos previstos para os dados de entrada. |

**Exceções**

| Exceção                                                                 | Descrição                                                                                                                            |
|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `TypeError`                                                             | Se X não for ndarray ou list.                                                                                                        |
| `ValueError`                                                            | Se X contiver valores diferentes de (0 e 1) ou (True e False), quando as características treinadas forem do tipo `'binary-features'` |
| [`FeatureDimensionMismatch`](../exceptions.md#featuredimensionmismatch) | Se o número de dimensões em X não corresponder ao esperado.                                                                          |
| [`ModelNotFittedError`](../exceptions.md#modelnotfittederror)           | Se o modelo ainda não tiver sido treinado e não possuir o conjunto de células de memória.                                            |

---

### update_clusters

```python
def update_clusters(self, mst_inconsistency_factor: Optional[float] = None):
    ...
```

Agrupa os clusters com base no fator de inconsistência da MST.

Utiliza a Árvore Geradora Mínima (MST) criada a partir da população de anticorpos para redefinir os clusters.
As arestas cujos pesos excedem a média somada ao valor de `mst_inconsistency_factor` multiplicado pelo desvio padrão
dos pesos das arestas são removidas. Cada grafo conectado após essa poda é tratado como um grupo distinto.

**Parâmetros**

| Nome                       | Tipo    | Padrão | Descrição                                      |
|----------------------------|---------|:------:|------------------------------------------------|
| `mst_inconsistency_factor` | `float` | `None` | Sobrescrever o fator de inconsistência da MST. |

**Exceções**

| Exceção      | Descrição                                                                     |
|--------------|-------------------------------------------------------------------------------|
| `ValueError` | Se a Árvore Geradora Mínima (MST) ainda não tiver sido criada.                |
| `ValueError` | Se a população de anticorpos estiver vazia.                                   |
| `ValueError` | Se as estatísticas da MST (média ou desvio padrão) não estiverem disponíveis. |

**Updates**

| Nome             | Tipo                     | Descrição                                                     |
|------------------|--------------------------|---------------------------------------------------------------|
| `memory_network` | `dict[int, npt.NDArray]` | Dicionário de rótulos de clusters para vetores de anticorpos. |
| `labels`         | `list`                   | Lista de rótulos de clusters.                                 |

---

## Exemplos Estendidos

Exemplos completos de uso estão disponíveis nos notebooks Jupyter:

- [**Random Example**](../../../../examples/en/clustering/AiNet/example_with_randomly_generated_dataset.ipynb)
- [**Geyser Dataset Example**](../../../../examples/en/clustering/AiNet/geyser_dataBase_example.ipynb)

---

## Referências

[^1]: De Castro, Leandro & José, Fernando & von Zuben, Antonio Augusto. (2001). aiNet: An Artificial Immune Network for
    Data Analysis.
    Available at: https://www.researchgate.net/publication/228378350_aiNet_An_Artificial_Immune_Network_for_Data_Analysis

[^2]: SciPy Documentation. *Minimum Spanning Tree*.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree
