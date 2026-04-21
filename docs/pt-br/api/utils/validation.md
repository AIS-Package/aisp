---
id: validation
sidebar_label: validation
keywords:
    - validação
---

# Módulo

Contém funções responsáveis pela validação de tipos de dados.

> **Módulo:** `aisp.utils.validation`  
> **Importação:** `from aisp.utils import validation`

## Funções

### detect_vector_data_type

```python
def detect_vector_data_type(vector: npt.NDArray) -> FeatureType:
    ...
```

Detecta o tipo de dado em um determinado vetor.

Esta função analisa o vetor de entrada e classifica seus dados como um dos tipos suportados:

* **binário**: Valores booleanos (True/False) ou inteiro 0/1.
* **contínuo**: Valores float dentro do intervalo normalizado [0.0, 1.0].
* **intervalo**: Valores float fora do intervalo normalizado.

**Parâmetros**

| Nome     | Tipo          | Padrão | Descrição                                         |
|----------|---------------|:------:|---------------------------------------------------|
| `vector` | `npt.NDArray` |   -    | Um array contendo os dados a serem classificados. |

**Returns**

| Tipo                                    | Descrição                                                                                           |
|-----------------------------------------|-----------------------------------------------------------------------------------------------------|
| [`FeatureType`](./types.md#featuretype) | O tipo de dado detectado no vetor: "binary-features", "continuous-features", or " ranged-features". |

**Exceções**

| Exceção                | Descrição                                          |
|------------------------|----------------------------------------------------|
| `UnsupportedTypeError` | Se o vetor contiver um tipo de dado não suportado. |

---

### check_array_type

```python
def check_array_type(x, name: str = "X") -> npt.NDArray:
    ...
```

Garante que X seja um array numpy. Converte a partir de lista, se necessário.

**Parâmetros**

| Nome   | Tipo  | Padrão | Descrição                                                                            |
|--------|-------|:------:|--------------------------------------------------------------------------------------|
| `x`    | `Any` |   -    | Array contendo as amostras e suas características. Formato: (n_samples, n_features). |
| `name` | `str` | `'X'`  | Nome da variável usado em mensagens de erro.                                         |

**Returns**

| Tipo          | Descrição                       |
|---------------|---------------------------------|
| `npt.NDArray` | O array convertido ou validado. |

**Exceções**

| Exceção     | Descrição                       |
|-------------|---------------------------------|
| `TypeError` | Se x não for ndarray nem lista. |

---

### check_shape_match

```python
def check_shape_match(x: npt.NDArray, y: npt.NDArray):
    ...
```

Garante que X e y tenham dimensões compatíveis.

**Parâmetros**

| Nome | Tipo          | Padrão | Descrição                                                                            |
|------|---------------|:------:|--------------------------------------------------------------------------------------|
| `x`  | `npt.NDArray` |   -    | Array contendo as amostras e suas características. Formato: (n_samples, n_features). |
| `y`  | `npt.NDArray` |   -    | Array com as classes alvo de `x`, com formato (`n_samples`).                         |

**Exceções**

| Exceção     | Descrição                                 |
|-------------|-------------------------------------------|
| `TypeError` | Se x ou y tiverem formatos incompatíveis. |

---

### check_feature_dimension

```python
def check_feature_dimension(x: npt.NDArray, expected: int):
    ...
```

Garante que o array possui o número esperado de características (features).

**Parâmetros**

| Nome       | Tipo          | Padrão | Descrição                                                                                                       |
|------------|---------------|:------:|-----------------------------------------------------------------------------------------------------------------|
| `x`        | `npt.NDArray` |   -    | Array de entrada para predição, contendo as amostras e suas características. Formato:: (n_samples, n_features). |
| `expected` | `int`         |   -    | Número esperado de features por amostra (colunas em X).                                                         |

**Exceções**

| Exceção                                                                 | Descrição                                                  |
|-------------------------------------------------------------------------|------------------------------------------------------------|
| [`FeatureDimensionMismatch`](../exceptions.md#featuredimensionmismatch) | Se o número de features em X não corresponder ao esperado. |

---

### check_binary_array

```python
def check_binary_array(x: npt.NDArray):
    ...
```

Garante que X contenha apenas valores (0 e 1) or (`True` e `False`).

**Parâmetros**

| Nome | Tipo          | Padrão | Descrição                   |
|------|---------------|:------:|-----------------------------|
| `x`  | `npt.NDArray` |   -    | Array contendo as amostras. |

**Exceções**

| Exceção      | Descrição                                         |
|--------------|---------------------------------------------------|
| `ValueError` | Se o array contiver valores diferentes de 0 e 1.  |

---

### check_value_range

```python
def check_value_range(
    x: npt.NDArray,
    name: str = 'X',
    min_value: float = 0.0,
    max_value: float = 1.0
) -> None:
    ...
```

Garante que todos os valores do array x estejam dentro do intervalo.

**Parâmetros**

| Nome        | Tipo          | Padrão | Descrição                       |
|-------------|---------------|:------:|---------------------------------|
| `x`         | `npt.NDArray` |   -    | Array contendo as amostras.     |
| `name`      | `str`         | `'X'`  | Nome usado na mensagem de erro. |
| `min_value` | `float`       | `0.0`  | Valor mínimo permitido.         |
| `max_value` | `float`       | `1.0`  | Valor máximo permitido.         |

**Exceções**

| Exceção      | Descrição                                                    |
|--------------|--------------------------------------------------------------|
| `ValueError` | Se o array estiver fora do intervalo (min_value, max_value). |
