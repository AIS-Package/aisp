# Metrics

## def accuracy_score(...)

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

# Multiclass

## def slice_index_list_by_class(...)

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

# Sanitizers

## def sanitize_choice(...)

```python
def sanitize_choice(value: T, valid_choices: Iterable[T], default: T) -> T
```

A função ``sanitize_choice(...)``, retorna o valor se estiver presente no conjunto de opções válidas; caso contrário, retorna o valor padrão.

**Parameters:**
* ***value*** (``T``): O valor a ser verificado.
* ***valid_choices*** (``Iterable[T]``): Uma coleção de opções válidas.
* ***default***: O valor padrão a ser retornado se ``value`` não estiver em ``valid_choices``.


**Returns:**
* `T`: O valor original, se válido, ou o valor padrão, se não.

---

## def sanitize_param(...)

```python
def sanitize_param(value: T, default: T, condition: Callable[[T], bool]) -> T:
```

A função ``sanitize_param(...)``, retorna o valor se ele satisfizer a condição especificada; caso contrário, retorna o valor padrão.

**Parameters:**
* value (``T``): O valor a ser verificado.
* default (``T``): O valor padrão a ser retornado se a condição não for satisfeita.
* condition (``Callable[[T], bool]``): Uma função que recebe um valor e retorna um booleano, determinando se o valor é válido.


**Returns:**
* `T`: O valor original se a condição for satisfeita, ou o valor padrão se não for.

---

## def sanitize_seed(...)

```python
def sanitize_seed(seed: Any) -> Optional[int]:
```

A função ``sanitize_param(...)``, retorna a semente se for um inteiro não negativo; caso contrário, retorna Nenhum.

**Parameters:**
* seed (``Any``): O valor da seed a ser validado.

**Returns:**
* ``Optional[int]``: A seed original se for um inteiro não negativo, ou ``None`` se for inválido.