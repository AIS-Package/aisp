---
id: metrics
sidebar_label: metrics
keywords:
  - accuracy
  - score
---

# metrics

Funções utilitárias para medir acurácia e desempenho.

> **Módulo:** `aisp.utils.metrics`  
> **Importação:** `from aisp.utils import metrics`

## Funções

### accuracy_score

```python
def accuracy_score(
    y_true: Union[npt.NDArray, list],
    y_pred: Union[npt.NDArray, list]
) -> float:
    ...
```

Calcula a acurácia com base nos rótulos reais e previstos.

**Parâmetros**

| Nome     | Tipo                       | Padrão | Descrição                                                         |
|----------|----------------------------|:------:|-------------------------------------------------------------------|
| `y_true` | `Union[npt.NDArray, list]` |   -    | Rótulos reais (corretos). Devem ter o mesmo tamanho que `y_pred`. |
| `y_pred` | `Union[npt.NDArray, list]` |   -    | Rótulos previstos. Devem ter o mesmo tamanho que `y_true`.        |

**Returns**

| Tipo    | Descrição                                                          |
|---------|--------------------------------------------------------------------|
| `float` | A proporção de previsões corretas em relação ao total de previsões |

**Exceções**

| Exceção      | Descrição                                                                |
|--------------|--------------------------------------------------------------------------|
| `ValueError` | Se y_true ou y_pred estiverem vazios ou se não tiverem o mesmo tamanho.  |
