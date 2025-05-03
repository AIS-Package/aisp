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


# Distance

Funções utilitárias para distância normalizada entre matrizes com decoradores numba.

## def hamming(...)

```python
def hamming(u: npt.NDArray, v: npt.NDArray) -> np.float64:
```

Função para calcular a distância de Hamming normalizada entre dois pontos.

    
$((x₁ ≠ x₂) + (y₁ ≠ y₂) + ... + (yn ≠ yn)) / n$


**Parameters:**
* u (``npt.NDArray``): Coordenadas do primeiro ponto
* v (``npt.NDArray``): Coordenadas do segundo ponto.

**Returns:**
* Distância (``float``) entre os dois pontos.

---

## def euclidean(...)

```python
def euclidean(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> np.float64:
```

Função para calcular a distância euclidiana normalizada entre dois pontos.

$√( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²)$



**Parameters:**
* u (``npt.NDArray``): Coordenadas do primeiro ponto
* v (``npt.NDArray``): Coordenadas do segundo ponto.

**Returns:**
* Distância (``float``) entre os dois pontos.

---

## def cityblock(...)

```python
def cityblock(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> np.float64:
```

Função para calcular a distância Manhattan normalizada entre dois pontos.
    
$(|x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) / n$


**Parameters:**
* u (``npt.NDArray``): Coordenadas do primeiro ponto
* v (``npt.NDArray``): Coordenadas do segundo ponto.

**Returns:**
* Distância (``float``) entre os dois pontos.

---

## def minkowski(...)

```python
def minkowski(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64], p: float = 2.0):
```

Função para calcular a distância de Minkowski normalizada entre dois pontos.
    
$(( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ) / n$


**Parameters:**
* u (``npt.NDArray``): Coordenadas do primeiro ponto.
* v (``npt.NDArray``): Coordenadas do segundo ponto.
* p (``float``): O parâmetro p define o tipo de distância a ser calculada:
    - p = 1: Distância **Manhattan** — soma das diferenças absolutas.
    - p = 2: Distância **Euclidiana** — soma das diferenças ao quadrado (raiz quadrada).
    - p > 2: Distância **Minkowski** com uma penalidade crescente à medida que p aumenta.

**Returns:**
* Distância (``float``) entre os dois pontos.

---

## def compute_metric_distance(...)

```python
def compute_metric_distance(
    u: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    metric: int,
    p: np.float64 = 2.0
) -> np.float64:
```

Função para calcular a distância entre dois pontos pela ``métrica`` escolhida.


**Parameters:**
* u (``npt.NDArray``): Coordenadas do primeiro ponto.
* v (``npt.NDArray``): Coordenadas do segundo ponto.
* metric (``int``): Métrica de distância a ser utilizada. Opções disponíveis: [0 (Euclidean), 1 (Manhattan), 2 (Minkowski)].
* p (``float``): Parâmetro da métrica de Minkowski (utilizado apenas se `metric` for "minkowski").
    

**Returns:**
* Distância (``double``) entre os dois pontos com a métrica selecionada.

---

## def min_distance_to_class_vectors(...)

```python
def min_distance_to_class_vectors(
    x_class: npt.NDArray,
    vector_x: npt.NDArray,
    metric: int,
    p: float = 2.0
) -> float:
```

Calcula a menor distância entre um vetor de entrada e os vetores de uma classe.

**Parameters:**
* x_class (``npt.NDArray``): Array contendo os vetores da classe com os quais o vetor de entrada será comparado. Formato esperado: (n_amostras, n_características).
* vector_x (``npt.NDArray``): Vetor a ser comparado com os vetores da classe. Formato esperado: (n_características,).
* metric (``int``): Métrica de distância a ser utilizada. Opções disponíveis: [0 (Euclidean), 1 (Manhattan), 2 (Minkowski)].
* p (``float``): Parâmetro da métrica de Minkowski (utilizado apenas se `metric` for "minkowski").

**Returns:**
* float: A menor distância calculada entre o vetor de entrada e os vetores da classe.
* Retorna -1.0 se as dimensões de entrada forem incompatíveis.

---

## def get_metric_code(...)

```python
def get_metric_code(metric: str) -> int:
```
Retorna o código numérico associado a uma métrica de distância.
  
**Parameters:**
* metric (``str``): Nome da métrica. Pode ser "euclidean", "manhattan", "minkowski" ou "hamming".


**Raises**
----------
* ``ValueError``: Se a métrica informada não for suportada.

**Returns:**
* ``int``: Código numérico correspondente à métrica.
