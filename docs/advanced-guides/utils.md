## Português

### Metrics

#### def accuracy_score(...)

```python
def accuracy_score(
    y_true: Union[npt.NDArray, list],
    y_pred: Union[npt.NDArray, list]
) -> float
```
Função para calcular a acurácia de precisão com base em listas de rótulos
verdadeiros e nos rótulos previstos.

**Parâmetros:**
* **_y_true_** (``Union[npt.NDArray, list]``): Rótulos verdadeiros (corretos)..
* **_y_pred_** (``Union[npt.NDArray, list]``): Rótulos previstos.

**Retorna:**
* **_Precisão_** (``float``): A proporção de previsões corretas em relação
    ao número total de previsões.

**Lança:**
* `ValueError`: Se `y_true` ou `y_pred` estiverem vazios ou se não
    tiverem o mesmo tamanho.

---

### Multiclass

#### def slice_index_list_by_class(...)

```python
def slice_index_list_by_class(classes, y: npt.NDArray) -> dict
```

A função ``__slice_index_list_by_class(...)``, separa os índices das linhas conforme a
classe de saída, para percorrer o array de amostra, apenas nas posições que a saída for
a classe que está sendo treinada.

**Parameters:**
* **_y_** (`npt.NDArray`): Recebe um array ``y``[``N amostra``] com as classes de saída do
    array de amostra ``X``.

**Returns:**
* `dict`: Um dicionário com a lista de posições do array(``y``), com as classes como chave.

## English

### Metrics

#### def accuracy_score(...)

```python
def accuracy_score(
        y_true: Union[npt.NDArray, list],
        y_pred: Union[npt.NDArray, list]
) -> float
```

Function to calculate precision accuracy based on lists of true labels and
predicted labels.

**Parameters**:
* **_y_true_** (``Union[npt.NDArray, list]``): Ground truth (correct) labels.
    Expected to be of the same length as `y_pred`.
* **_y_pred_** (``Union[npt.NDArray, list]``): Predicted labels. Expected to
    be of the same length as `y_true`.

Returns:
* **_Accuracy_** (``float``): The ratio of correct predictions to the total
number of predictions.

**Raises**:
* `ValueError`: If `y_true` or `y_pred` are empty or if they do not have the same length.

---

### Multiclass

#### def slice_index_list_by_class(...)

```python
def slice_index_list_by_class(classes, y: npt.NDArray) -> dict
```

The function ``__slice_index_list_by_class(...)``, separates the indices of the lines
according to the output class, to loop through the sample array, only in positions where
the output is the class being trained.

**Parameters:**
* **_y_** (npt.NDArray): Receives a ``y``[``N sample``] array with the output classes of the
    ``X`` sample array.

**returns:**
* `dict`: A dictionary with the list of array positions(``y``), with the classes as key.