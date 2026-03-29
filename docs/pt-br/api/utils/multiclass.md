---
id: multiclass
sidebar_label: multiclass
keywords:
    - k-nearest neighbors
    - samples
    - slice
    - multiclass
---

# multiclass

Funções utilitárias para lidar com dados com múltiplas classes.

> **Módulo:** `aisp.utils.multiclass`  
> **Importação:** `from aisp.utils import multiclass`

## Funções

### slice_index_list_by_class

```python
def slice_index_list_by_class(classes: Optional[Union[npt.NDArray, list]], y: npt.NDArray) -> dict:
    ...
```

Separa os índices das amostras por classe para iteração direcionada.

**Parâmetros**

| Nome      | Tipo                                 | Padrão | Descrição                                                                         |
|-----------|--------------------------------------|:------:|-----------------------------------------------------------------------------------|
| `classes` | `Optional[Union[npt.NDArray, list]]` |   -    | lista com classes únicas.                                                         |
| `y`       | `npt.NDArray`                        |   -    | Recebe um array `y` (`n_samples`) om as classes de saída do array de amostra `X`. |

**Returns**

| Tipo   | Descrição                                                                       |
|--------|---------------------------------------------------------------------------------|
| `dict` | Um dicionário com a lista de posições do array(`y`), com as classes como chave. |

**Example**

```python
import numpy as np
from aisp.utils.multiclass import slice_index_list_by_class

labels = ['a', 'b', 'c']
y = np.array(['a', 'c', 'b', 'a', 'c', 'b'])
slice_index_list_by_class(labels, y)
```

---

### predict_knn_affinity

```python
def predict_knn_affinity(
    X: npt.NDArray,
    k: int,
    all_cell_vectors: List[Tuple[Union[str, int], npt.NDArray]],
    affinity_func: Callable[[npt.NDArray, npt.NDArray], float]
) -> npt.NDArray:
    ...
```

Prever classes usando k-vizinhos mais próximos e células treinadas.

**Parâmetros**

| Nome               | Tipo                                          | Padrão | Descrição                                                       |
|--------------------|-----------------------------------------------|:------:|-----------------------------------------------------------------|
| `X`                | `npt.NDArray`                                 |   -    | Dados de entrada a serem classificados.                         |
| `k`                | `int`                                         |   -    | Número de vizinhos mais próximos a considerar para a previsão.  |
| `all_cell_vectors` | `List[Tuple[Union[str, int], npt.NDArray]]`   |   -    | Lista de tuplas contendo (nome_da_classe, cell(np.ndarray)).    |
| `affinity_func`    | `Callable[[npt.NDArray, npt.NDArray], float]` |   -    | Função que recebe dois vetores e retorna um valor de afinidade. |

**Returns**

| Tipo          | Descrição                                                                                |
|---------------|------------------------------------------------------------------------------------------|
| `npt.NDArray` | Array de rótulos previstos para cada amostra em X, baseado nos k vizinhos mais próximos. |
