---
id: rnsa
sidebar_label: RNSA
keywords:
  - seleção negativa
  - detecção de anomalias
  - reconhecimento do não-próprio
  - reconhecimento de padrões
  - classification
  - real-valued
  - v-detector
  - multiclass
tags:
  - classification
  - supervised
  - negative selection
  - real-valued
  - anomaly detection
---

# RNSA

Algoritmo de Seleção Negativa com valores reais (RNSA).

:::tip[Herança]

Esta classe herda de [BaseClassifier](../base/base-classifier.md)

:::


> **Módulo:** `aisp.nsa`  
> **Importação:** `from aisp.nsa import RNSA`

---

## Visão geral

Algoritmo para classificação e detecção de anomalias baseado na distinção entre próprio e não-próprio, inspirado
no algoritmo de seleção negativa.

:::note

Este algoritmo possui duas versões diferentes: uma baseada na versão canônica [^1] e outra com detectores com raio
variável [^2]. Ambas estão adaptadas para trabalhar com múltiplas classes e possuem métodos para previsão de
dados presentes na região não-self de todos os detectores e classes.

:::

:::warning

Os parâmetros `r` e `r_s` podem impedir a geração de detectores válidos. Um valor para `r` muito pequeno pode limitar
a cobertura, enquanto muito alto pode dificultar a geração de detectores validos. Da mesma forma, um `r_s` alto
pode limitar a criação de detectores. Portanto, o ajuste adequado de `r` e `r_s` é essencial para garantir um bom
desempenho do algoritmo.

:::

---

## Exemplo

```python
import numpy as np
from aisp.nsa import RNSA

np.random.seed(1)
class_a = np.random.uniform(high=0.5, size=(50, 2))
class_b = np.random.uniform(low=0.51, size=(50, 2))
```

**Exemplo 1:** Classificação multiclasse (RNSA suporta duas ou mais classes)

```python
x_train = np.vstack((class_a, class_b))
y_train = ['a'] * 50 + ['b'] * 50
rnsa = RNSA(N=150, r=0.3, seed=1)
rnsa = rnsa.fit(x_train, y_train, verbose=False)
x_test = [
    [0.15, 0.45],  # Expected: Class 'a'
    [0.85, 0.65],  # Expected: Class 'b'
]
y_pred = rnsa.predict(x_test)
print(y_pred)
```

**Output**

```bash
['a' 'b']
```

**Exemplo 2:** Detecção de anomalias (self/non-self)

```python
rnsa = RNSA(N=150, r=0.3, seed=1)
rnsa = rnsa.fit(X=class_a, y=np.array(['self'] * 50), verbose=False)
y_pred = rnsa.predict(class_b[:5])
print(y_pred)
```

**Output**

```bash
['non-self' 'non-self' 'non-self' 'non-self' 'non-self']
```

---

## Parâmetros do Construtor

| Nome             | Tipo                                      |     Default     | Descrição                                                                                                                                                                                                                                                            |
|------------------|-------------------------------------------|:---------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `N`              | `int`                                     |      `100`      | Quantidade de detectores.                                                                                                                                                                                                                                            |
| `r`              | `float`                                   |     `0.05`      | Raio do detector.                                                                                                                                                                                                                                                    |
| `r_s`            | `float`                                   |    `0.0001`     | O valor de rₛ é o raio das amostras próprias dos dados de treinamento X.                                                                                                                                                                                             |
| `k`              | `int`                                     |       `1`       | Quantidade de vizinhos próximos dos detectores gerados aleatoriamente para efetuar o cálculo da média da distância.                                                                                                                                                  |
| `metric`         | `{"euclidean", "minkowski", "manhattan"}` |  `'euclidean'`  | Métrica de distância usada para calcular a distância entre o detector e a amostra.                                                                                                                                                                                   |
| `max_discards`   | `int`                                     |     `1000`      | Número máximo de descartes de detectores em sequência, que tem como objetivo evitar um possível loop infinito caso seja definido um raio que não seja possível gerar detectores do não-próprio.                                                                      |
| `seed`           | `Optional[int]`                           |     `None`      | Seed para geração aleatória.                                                                                                                                                                                                                                         |
| `algorithm`      | `{"default-NSA", "V-detector"}`           | `'default-NSA'` | Define a versão do algoritmo.                                                                                                                                                                                                                                        |
| `non_self_label` | `str`                                     |  `'non-self'`   | Rótulo atribuído quando há apenas uma classe de saída e a amostra não pertence a essa classe.                                                                                                                                                                        |
| `cell_bounds`    | `bool`                                    |     `False`     | Se definido como True, esta opção limita a geração dos detectores ao espaço do plano compreendido entre 0 e 1. Isso significa que qualquer detector cujo raio ultrapasse esse limite é descartado, e esta variável é usada exclusivamente no algoritmo `V-detector`. |
| `p`              | `bool`                                    |      `2.0`      | Valor de `p` utilizado na distância de Minkowski.                                                                                                                                                                                                                    |

## Atributos

| Nome        | Tipo                                         | Padrão | Descrição                                       |
|-------------|----------------------------------------------|:------:|-------------------------------------------------|
| `detectors` | `Optional[Dict[str \| int, list[Detector]]]` |   -    | Conjunto de detectores, organizados por classe. |

---

## Métodos Públicos

### fit

```python
def fit(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list],
    verbose: bool = True,
) -> BNSA:
    ...
```

Treinamento de acordo com X e y, utilizando o algoritmo de seleção negativa.

**Parâmetros**

| Nome      | Tipo                       | Padrão | Descrição                                                                                      |
|-----------|----------------------------|:------:|------------------------------------------------------------------------------------------------|
| `X`       | `Union[npt.NDArray, list]` |   -    | Dados de treinamento contendo as amostras e com suas características. (n_samples, n_features). |
| `y`       | `Union[npt.NDArray, list]` |   -    | Array com as classes alvo de `X`, com dimensão (n_samples).                                    |
| `verbose` | `bool`                     | `True` | Indica se as mensagens de progresso de geração dos detectores.                                 |

**Returns**

| Tipo   | Descrição                      |
|--------|--------------------------------|
| `Self` | Retorna a instancia da classe. |

**Exceções**

| Exceção                                                               | Descrição                                                                                                                   |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `TypeError`                                                           | Se X ou y não forem ndarrays ou tiverem tamanhos incompatíveis.                                                             |
| `ValueError`                                                          | Se os valores de X estiverem fora do intervalo (0.0, 1.0).                                                                  |
| [`MaxDiscardsReachedError`](../exceptions.md#maxdiscardsreachederror) | Se o número máximo de descartes for atingido durante a maturação. Verifique o valor do raio definido e considere reduzi-lo. |

---

### predict

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
    ...
```

Prever as classes com base nos detectores gerados após o treinamento.

**Parâmetros**

| Nome | Tipo                       | Padrão | Descrição                                                       |
|------|----------------------------|:------:|-----------------------------------------------------------------|
| `X`  | `Union[npt.NDArray, list]` |   -    | Array com amostras de entrada no com: (`n_samples, n_features`) |

**Returns**

| Tipo          | Descrição                                                       |
|---------------|-----------------------------------------------------------------|
| `npt.NDArray` | Array `C` (`n_samples`), contendo as classes prevista para `X`. |

**Exceções**


| Exceção                                                                 | Descrição                                                                                |
|-------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| `TypeError`                                                             | Se X não for ndarray ou list.                                                            |
| `ValueError`                                                            | Se os valores de X estiverem fora do intervalo (0.0, 1.0).                               |
| [`FeatureDimensionMismatch`](../exceptions.md#featuredimensionmismatch) | Se o número de característica (colunas) em X não corresponder ao valor esperado.         |
| [`ModelNotFittedError`](../exceptions.md#modelnotfittederror)           | Se o modelo ainda não tiver sido treinado e não possuir detectores ou classes definidas. |

---

## Exemplos Estendidos

Exemplos completos de uso estão disponíveis nos notebooks Jupyter:

- [**Iris Dataset Example**](../../../../examples/en/classification/RNSA/iris_dataBase_example_en.ipynb)
- [**Geyser Dataset Example**](../../../../examples/en/classification/RNSA/geyser_dataBase_example_en.ipynb)
- [**Random Dataset Example**](../../../../examples/en/classification/RNSA/example_with_randomly_generated_dataset-en.ipynb)

---

## Referências

[^1]: BRABAZON, Anthony; O'NEILL, Michael; MCGARRAGHY, Seán. Natural Computing
    Algorithms. [S. l.]: Springer Berlin Heidelberg, 2015. DOI 10.1007/978-3-662-43631-8.
    Disponível em: https://dx.doi.org/10.1007/978-3-662-43631-8.

[^2] Ji, Z.; Dasgupta, D. (2004). Real-Valued Negative Selection Algorithm with Variable-Sized Detectors.
    In *Lecture Notes in Computer Science*, vol. 3025. https://doi.org/10.1007/978-3-540-24854-5_30
