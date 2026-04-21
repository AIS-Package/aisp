---
id: distance
sidebar_label: distance
keywords:
  - hamming
  - euclidean
  - cityblock
  - Manhattan
  - minkowski
  - distância
---

# distance

Funções utilitárias para cálculo de distância entre vetores com decoradores numba.

> **Módulo:** `aisp.utils.distance`  
> **Importação:** `from aisp.utils import distance`

## Funções

### hamming

```python
@njit([(types.boolean[:], types.boolean[:])], cache=True)
def hamming(u: npt.NDArray[np.bool_], v: npt.NDArray[np.bool_]) -> float64:
    ...
```

Calcula a distância de Hamming entre dois pontos.

$$
\frac{(x_1 \neq y_1) + (x_2 \neq y_2) + \cdots + (x_n \neq y_n)}{n}
$$

**Parâmetros**

| Nome | Tipo                    | Padrão | Descrição                      |
|------|-------------------------|:------:|--------------------------------|
| `u`  | `npt.NDArray[np.bool_]` |   -    | Coordenadas do primeiro ponto. |
| `v`  | `npt.NDArray[np.bool_]` |   -    | Coordenadas do segundo ponto.  |

**Returns**

| Tipo      | Descrição                               |
|-----------|-----------------------------------------|
| `float64` | Distância de Hamming entre dois pontos. |

---

### euclidean

```python
@njit()
def euclidean(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float64:
    ...
```

Calcula a distância de Euclidean entre dois pontos.

$$
\sqrt{(X_{1} - X_{1})^2 + (Y_{2} - Y_{2})^2 + \cdots + (Y_{n} - Y_{n})^2}
$$

**Parâmetros**

| Nome | Tipo                   | Padrão | Descrição                      |
|------|------------------------|:------:|--------------------------------|
| `u`  | `npt.NDArray[float64]` |   -    | Coordenadas do primeiro ponto. |
| `v`  | `npt.NDArray[float64]` |   -    | Coordenadas do segundo ponto.  |

**Returns**

| Tipo      | Descrição                                 |
|-----------|-------------------------------------------|
| `float64` | Distância de Euclidean entre dois pontos. |

---

### cityblock

```python
@njit()
def cityblock(u: npt.NDArray[float64], v: npt.NDArray[float64]) -> float64:
    ...
```

Calcula a distância de Manhattan entre dois pontos.

$$
|X_{1} - Y_{1}| + |X_{2} - Y_{2}| + \cdots + |X_{n} - Y_{n}|
$$

**Parâmetros**

| Nome | Tipo                   | Padrão | Descrição                      |
|------|------------------------|:------:|--------------------------------|
| `u`  | `npt.NDArray[float64]` |   -    | Coordenadas do primeiro ponto. |
| `v`  | `npt.NDArray[float64]` |   -    | Coordenadas do segundo ponto.  |

**Returns**

| Tipo      | Descrição                                 |
|-----------|-------------------------------------------|
| `float64` | Distância de Manhattan entre dois pontos. |

---

### minkowski

```python
@njit()
def minkowski(
    u: npt.NDArray[float64],
    v: npt.NDArray[float64],
    p: float = 2.0
) -> float64:
    ...
```

Calcula a distância de Minkowski entre dois pontos.

$$
(|X_{1} - Y_{1}|^p + |X_{2} - Y_{2}|^p + \cdots + |X_{n} - Y_{n}|^p)^\frac{1}{p}|
$$

**Parâmetros**

| Nome | Tipo                   | Padrão | Descrição                                                  |
|------|------------------------|:------:|------------------------------------------------------------|
| `u`  | `npt.NDArray[float64]` |   -    | Coordenadas do primeiro ponto.                             |
| `v`  | `npt.NDArray[float64]` |   -    | Coordenadas do segundo ponto.                              |
| `p`  | `float`                | `2.0`  | Parâmetro que define o tipo de distância a ser calculada.  |

:::note[Parâmetro `p`]

- p = 1: distância **Manhattan** - soma das diferenças
- p = 2: distância **Euclidean** - soma dos quadrados (raiz quadrada).
- p > 2: distância **Minkowski** com penalização crescente conforme `p` aumenta.

:::

**Returns**

| Tipo      | Descrição                                 |
|-----------|-------------------------------------------|
| `float64` | Distância de Minkowski entre dois pontos. |

---

### compute_metric_distance

```python
@njit([(types.float64[:], types.float64[:], types.int32, types.float64)], cache=True)
def compute_metric_distance(
    u: npt.NDArray[float64],
    v: npt.NDArray[float64],
    metric: int,
    p: float = 2.0
) -> float64:
    ...
```

Calcula a distância entre dois pontos com base na métrica escolhida.

**Parâmetros**

| Nome     | Tipo                   | Padrão | Descrição                                                                                |
|----------|------------------------|:------:|------------------------------------------------------------------------------------------|
| `u`      | `npt.NDArray[float64]` |   -    | Coordenadas do primeiro ponto.                                                           |
| `v`      | `npt.NDArray[float64]` |   -    | Coordenadas do segundo ponto.                                                            |
| `metric` | `int`                  |   -    | Métrica de distância que será usada. Opções: 0 (Euclidean), 1 (Manhattan), 2 (Minkowski) |
| `p`      | `float`                | `2.0`  | Parâmetro que define o tipo de distância a ser calculada                                 |

**Returns**

| Tipo      | Descrição                                                 |
|-----------|-----------------------------------------------------------|
| `float64` | Distância entre os dois pontos com a métrica selecionada. |

---

### min_distance_to_class_vectors

```python
@njit([(types.float64[:, :], types.float64[:], types.int32, types.float64)], cache=True)
def min_distance_to_class_vectors(
    x_class: npt.NDArray[float64],
    vector_x: npt.NDArray[float64],
    metric: int,
    p: float = 2.0,
) -> float:
    ...
```

Calcula distância entre um vetor de entrada e os vetores de uma classe e retorna a distância minima.

**Parâmetros**

| Nome       | Tipo                   | Padrão | Descrição                                                                                                                 |
|------------|------------------------|:------:|---------------------------------------------------------------------------------------------------------------------------|
| `x_class`  | `npt.NDArray[float64]` |   -    | Array contendo os vetores da classe a serem comparados com o vetor de entrada. Formato esperado: (n_samples, n_features). |
| `vector_x` | `npt.NDArray[float64]` |   -    | Vetor a ser comparado com os vetores da classe. Formato esperado: (n_features,).                                          |
| `metric`   | `int`                  |   -    | Métrica de distância a ser usada. Opções: 0 ("euclidean"), 1 ("manhattan"), 2 ("minkowski") or ("hamming")                |
| `p`        | `float`                | `2.0`  | Parâmetro da distância de Minkowski (usado apenas quando metric="minkowski").                                             |

**Returns**

| Tipo    | Descrição                                                                                                                      |
|---------|--------------------------------------------------------------------------------------------------------------------------------|
| `float` | A menor distância calculada entre o vetor de entrada e os vetores da classe. Retorna -1.0 se as dimensões forem incompatíveis. |

---

### get_metric_code

```python
def get_metric_code(metric: str) -> int:
    ...
```

Obtém o valor associado a uma métrica de distância.

**Parâmetros**

| Nome     | Tipo  | Padrão | Descrição                                                                             |
|----------|-------|:------:|---------------------------------------------------------------------------------------|
| `metric` | `str` |   -    | Nome da métrica. Pode ser `"euclidean"`, `"manhattan"`, `"minkowski"` or `"hamming"`. |

**Returns**

| Tipo  | Descrição                        |
|-------|----------------------------------|
| `int` | Número correspondente à métrica. |

**Exceções**

| Exceção      | Descrição                                 |
|--------------|-------------------------------------------|
| `ValueError` | Se a métrica fornecida não for suportada. |

