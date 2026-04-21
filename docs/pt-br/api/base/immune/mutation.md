---
id: mutation
sidebar_label: mutation
keywords:
  - mutações
  - expansão clonal
  - sistema imune
  - funções python com numba
  - vetor de mutações
---

# mutation

As funções utilizam decoradores do Numba para compilação Just-In-Time(JIT).

Contém funções que geram conjuntos de clones hipermutados a partir de vetores contínuos ou binários, simulando o
processo
de expansão clonal dos sistemas imunológicos artificiais.

> **Módulo:** `aisp.base.immune`  
> **Importação:** `from aisp.base.immune import mutation`

## Funções

### clone_and_mutate_continuous

```python
@njit([(types.float64[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_continuous(
    vector: npt.NDArray[np.float64],
    n: int,
    mutation_rate: float
) -> npt.NDArray[np.float64]:
    ...
```

Gera um conjunto de clones mutados a partir de um vetor contínuo.

Esta função cria `n` clones do vetor de entrada e aplica mutações em cada um, simulando o processo
de expansão clonal em sistemas imunes artificiais.

**Parâmetros**

| Nome            | Tipo                      | Padrão | Descrição                                                                                                                                    |
|-----------------|---------------------------|:------:|----------------------------------------------------------------------------------------------------------------------------------------------|
| `vector`        | `npt.NDArray[np.float64]` |   -    | Vetor contínuo original que representa a célula imune a ser clonada e mutada.                                                                |
| `n`             | `int`                     |   -    | Quantidade de clones mutados que serão gerados.                                                                                              |
| `mutation_rate` | `float`                   |   -    | Se 0 ≤ mutation_rate < 1, usa probabilidade de mutação por características. Caso contrario, um número aleatorio de características é mutado. |

**Returns**

| Tipo                      | Descrição                                                                              |
|---------------------------|----------------------------------------------------------------------------------------|
| `npt.NDArray[np.float64]` | Array com dimensões (n, len(vector)) contendo os `n` clones mutados do vetor original. |

### clone_and_mutate_binary

```python
@njit([(types.boolean[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_binary(
    vector: npt.NDArray[np.bool_],
    n: int,
    mutation_rate: float
) -> npt.NDArray[np.bool_]:
    ...
```

Gera um conjunto de clones mutados a partir de um vetor binário.

Esta função cria `n` clones do vetor binário de entrada e aplica mutações aos bits, simulando a expansão
clonal em sistemas imunes artificiais com representações discretas.

**Parâmetros**

| Nome            | Tipo                      | Padrão | Descrição                                                                                                                                    |
|-----------------|---------------------------|:------:|----------------------------------------------------------------------------------------------------------------------------------------------|
| `vector`        | `npt.NDArray[np.float64]` |   -    | Vetor binário original que representa a célula imune a ser clonada e mutada.                                                                 |
| `n`             | `int`                     |   -    | Quantidade de clones mutados a serão gerados.                                                                                                |
| `mutation_rate` | `float`                   |   -    | Se 0 ≤ mutation_rate < 1, usa probabilidade de mutação por características. Caso contrario, um número aleatorio de características é mutado. |

**Returns**

| Tipo                    | Descrição                                                                              |
|-------------------------|----------------------------------------------------------------------------------------|
| `npt.NDArray[np.bool_]` | Array com dimensões (n, len(vector)) contendo os `n` clones mutados do vetor original. |

### clone_and_mutate_ranged

```python
@njit([(types.float64[:], types.int64, types.float64[:, :], types.float64)], cache=True)
def clone_and_mutate_ranged(
    vector: npt.NDArray[np.float64],
    n: int,
    bounds: npt.NDArray[np.float64],
    mutation_rate: float,
) -> npt.NDArray[np.float64]:
    ...
```

Gera um conjunto de clones mutados a partir de uma célula representada pelo intervalo personalizados por dimensão..

Esta função cria `n` clones do vetor de entrada e aplica mutações em cada um, simulando o processo
de expansão clonal em sistemas imunes artificiais.

**Parâmetros**

| Nome            | Tipo                      | Padrão | Descrição                                                                                                                                    |
|-----------------|---------------------------|:------:|----------------------------------------------------------------------------------------------------------------------------------------------|
| `vector`        | `npt.NDArray[np.float64]` |   -    | Vetor contínuo original que representa a célula imune a ser clonada e mutada.                                                                |
| `n`             | `int`                     |   -    | Quantidade de clones mutados a serão gerados.                                                                                                |
| `bounds`        | `np.ndarray`              |   -    | Array (n_features, 2) com valor mínimo e máximo por dimensão.                                                                                |
| `mutation_rate` | `float`                   |   -    | Se 0 ≤ mutation_rate < 1, usa probabilidade de mutação por características. Caso contrario, um número aleatorio de características é mutado. |

**Returns**

| Tipo                      | Descrição                                                                              |
|---------------------------|----------------------------------------------------------------------------------------|
| `npt.NDArray[np.float64]` | Array com dimensões (n, len(vector)) contendo os `n` clones mutados do vetor original. |

### clone_and_mutate_continuous

```python
@njit([(types.int64[:], types.int64, types.float64)], cache=True)
def clone_and_mutate_permutation(
    vector: npt.NDArray[np.int64],
    n: int,
    mutation_rate: float
) -> npt.NDArray[np.int64]:
    ...
```

Gera um conjunto de clones com mutações por permutação.

**Parâmetros**

| Nome            | Tipo                    | Padrão | Descrição                                                                          |
|-----------------|-------------------------|:------:|------------------------------------------------------------------------------------|
| `vector`        | `npt.NDArray[np.int64]` |   -    | Vetor de permutação original que representa a célula imune a ser clonada e mutada. |
| `n`             | `int`                   |   -    | Quantidade de clones mutados a serão gerados.                                      |
| `mutation_rate` | `float`                 |   -    | Probabilidade de mutação de uma característica.                                    |

**Returns**

| Tipo                      | Descrição                                                                              |
|---------------------------|----------------------------------------------------------------------------------------|
| `npt.NDArray[np.float64]` | Array com dimensões (n, len(vector)) contendo os `n` clones mutados do vetor original. |
