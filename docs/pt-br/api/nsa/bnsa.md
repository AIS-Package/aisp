---
id: bnsa
sidebar_label: BNSA
keywords:
  - seleção negativa
  - características binárias
  - detecção de anomalias
  - reconhecimento do não-próprio
  - reconhecimento de padrões
  - classificação
  - multiclasses
tags:
  - classificação
  - supervisionado
  - seleção negativa
  - características binárias
  - detecção de anomalias
---

# BNSA

Algoritmo de Seleção Negativa Binária (BNSA).

:::tip[Herança]
Esta classe herda de [BaseClassifier](../base/base-classifier.md)
:::


> **Módulo:** `aisp.nsa`  
> **Importação:** `from aisp.nsa import BNSA`

---

## Visão geral

Algoritmo para classificação e detecção de anomalias baseado na distinção entre próprio e não-próprio, inspirado
no algoritmo de seleção negativa.

:::note

O _**Binary Negative Selection Algorithm (BNSA)**_ é baseado na proposta original de Forrest et al. (1994) [^1],
desenvolvido para segurança na computação. Nesta adaptação, o algoritmo usa arrays de bits e possuir suporte para
classificação multiclasse.

:::

:::warning

Valores altos de `aff_thresh` podem impedir que gere detectores válidos para a detecção do não-próprio.

:::

---

## Exemplo

```python
from aisp.nsa import BNSA

# Binary 'self' samples
x_train = [
    [0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 1, 0]
]
y_train = ['self', 'self', 'self', 'self', 'self', 'self']
bnsa = BNSA(aff_thresh=0.55, seed=1)
bnsa = bnsa.fit(x_train, y_train, verbose=False)
# samples for testing
x_test = [
    ...[1, 1, 1, 1, 1],  # Sample of Anomaly
    ...[0, 1, 0, 1, 0],  # self sample
    ...]
y_prev = bnsa.predict(X=x_test)
print(y_prev)
```

**Output**

```bash
['non-self' 'self']
```

---

## Parâmetros do Construtor

| Nome                        | Tipo            |          Default           | Descrição                                                                                                                                                                                   |
|-----------------------------|-----------------|:--------------------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `N`                         | `int`           |           `100`            | Quantidade de detectores.                                                                                                                                                                   |
| `aff_thresh`                | `float`         |           `0.1`            | Representa a porcentagem de similaridade entre a célula T e as amostras próprias. O valor padrão é de 10% (0,1), enquanto que o valor de 1,0 representa 100% de similaridade.               |
| `max_discards`              | `int`           |           `1000`           | Número máximo de descartes de detectores em sequência, com o objetivo de evitar um possível loop infinito caso seja definido um raio que não seja possível gerar detectores do não-próprio. |
| `seed`                      | `Optional[int]` |           `None`           | Seed para geração aleatória.                                                                                                                                                                |
| `no_label_sample_selection` | `str`           | `'max_average_difference'` | Método utilizado para a escolha de rótulos para amostras classificada como não-próprio por todos os detectores.                                                                             |

## Atributos

| Nome        | Tipo                                                | Padrão | Descrição                                       |
|-------------|-----------------------------------------------------|:------:|-------------------------------------------------|
| `detectors` | `Optional[Dict[str \| int, npt.NDArray[np.bool_]]]` |   -    | Conjunto de detectores, organizados por classe. |

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
| `ValueError`                                                          | Se X possuir valores diferentes de (0 e 1) ou (True e False).                                                               |
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
| `ValueError`                                                            | Se X possuir valores diferentes de (0 e 1) ou (True e False).                            |
| [`FeatureDimensionMismatch`](../exceptions.md#featuredimensionmismatch) | Se o número de característica (colunas) em X não corresponder ao valor esperado.         |
| [`ModelNotFittedError`](../exceptions.md#modelnotfittederror)           | Se o modelo ainda não tiver sido treinado e não possuir detectores ou classes definidas. |

---

## Exemplos Estendidos

Exemplos completos de uso estão disponíveis nos notebooks Jupyter:

- [**Mushrooms Dataset Example**](../../../../examples/en/classification/BNSA/mushrooms_dataBase_example_en.ipynb)
- [**Random Dataset Example**](../../../../examples/en/classification/BNSA/example_with_randomly_generated_dataset-en.ipynb)

---

## Referências

[^1]: S. Forrest, A. S. Perelson, L. Allen and R. Cherukuri, "Self-nonself discrimination in
    a computer," Proceedings of 1994 IEEE Computer Society Symposium on Research in Security
    and Privacy, Oakland, CA, USA, 1994, pp. 202-212,
    doi: https://dx.doi.org/10.1109/RISP.1994.296580.
