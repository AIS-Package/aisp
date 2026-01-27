# Utils

Funções de utilidade para o desenvolvimento.

## Métricas (Metrics)

### Função `accuracy_score(...)`

```python
def accuracy_score(
    y_true: Union[npt.NDArray, list],
    y_pred: Union[npt.NDArray, list]
) -> float:
```

Função para calcular a acurácia de precisão com base em listas de rótulos
verdadeiros e nos rótulos previstos.

**Parâmetros:**

* **y_true** (``Union[npt.NDArray, list]``): Rótulos verdadeiros (corretos)..
* **y_pred** (``Union[npt.NDArray, list]``): Rótulos previstos.

**Retorna:**

* **Precisão** (``float``): A proporção de previsões corretas em relação
    ao número total de previsões.

**Exceções:**

* `ValueError`: Se `y_true` ou `y_pred` estiverem vazios ou se não
    tiverem o mesmo tamanho.

---

## Multiclass

### Função `predict_knn_affinity(...)`

```python
def predict_knn_affinity(
    X: npt.NDArray,
    k: int,
    all_cell_vectors: List[Tuple[Union[str, int], npt.NDArray]],
    affinity_func: Callable[[npt.NDArray, npt.NDArray], float]
) -> npt.NDArray
```

Função para prever classes usando k-vizinhos mais próximos e células treinadas.

**Parâmetros:**

* **X** (`npt.NDArray`): Dados de entrada a serem classificados.
* **k** (`int`): Número de vizinhos mais próximos a considerar para a previsão.
* **all_cell_vectors** (`List[Tuple[Union[str, int], npt.NDArray]]`): Lista de tuplas contendo pares (nome_da_classe, vetor_da_célula).
* **affinity_func** (`Callable[[npt.NDArray, npt.NDArray], float]`): Função que recebe dois vetores e retorna um valor de afinidade.

**Retorna:**

* `npt.NDArray`: Array de rótulos previstos para cada amostra em X, baseado nos k vizinhos mais próximos.

---

### Função `slice_index_list_by_class(...)`

```python
def slice_index_list_by_class(classes: Optional[Union[npt.NDArray, list]], y: npt.NDArray) -> dict
```

A função ``slice_index_list_by_class(...)``, separa os índices das amostras
conforme a classe de saída, para percorrer o array de amostra apenas nas posições
onde a saída corresponde à classe sendo treinada.

**Parâmetros:**

* **classes** (`Optional[Union[npt.NDArray, list]]`): Lista com classes únicas. Se None, retorna um dicionário vazio.
* **y** (`npt.NDArray`): Array com as classes de saída do array de amostra ``X``.

**Retorna:**

* `dict`: Um dicionário com a lista de posições do array, com as classes como chave.

**Exemplos:**

```python
>>> import numpy as np
>>> labels = ['a', 'b', 'c']
>>> y = np.array(['a', 'c', 'b', 'a', 'c', 'b'])
>>> slice_index_list_by_class(labels, y)
{'a': [0, 3], 'b': [2, 5], 'c': [1, 4]}
```

## Sanitizers

### Função `sanitize_choice(...)`

```python
def sanitize_choice(value: T, valid_choices: Iterable[T], default: T) -> T
```

A função ``sanitize_choice(...)``, retorna o valor se estiver presente no conjunto de opções válidas; caso contrário, retorna o valor padrão.

**Parâmetros:**

* **value** (``T``): O valor a ser verificado.
* **valid_choices** (``Iterable[T]``): Uma coleção de opções válidas.
* **default**: O valor padrão a ser retornado se ``value`` não estiver em ``valid_choices``.

**Returns:**

* `T`: O valor original, se válido, ou o valor padrão, se não.

---

### Função `sanitize_param(...)`

```python
def sanitize_param(value: T, default: T, condition: Callable[[T], bool]) -> T:
```

A função ``sanitize_param(...)``, retorna o valor se ele satisfizer a condição especificada; caso contrário, retorna o valor padrão.

**Parâmetros:**

* **value** (``T``): O valor a ser verificado.
* **default** (``T``): O valor padrão a ser retornado se a condição não for satisfeita.
* **condition** (``Callable[[T], bool]``): Uma função que recebe um valor e retorna um booleano, determinando se o valor é válido.

**Returns:**

* `T`: O valor original se a condição for satisfeita, ou o valor padrão se não for.

---

### Função `sanitize_seed(...)`

```python
def sanitize_seed(seed: Any) -> Optional[int]:
```

A função ``sanitize_seed(...)``, retorna a semente se for um inteiro não negativo; caso contrário, retorna Nenhum.

**Parâmetros:**

* **seed** (``Any``): O valor da seed a ser validado.

**Returns:**

* ``Optional[int]``: A seed original se for um inteiro não negativo, ou ``None`` se for inválido.

---

### Função `sanitize_bounds(...)`

```python
def sanitize_bounds(bounds: Any, problem_size: int) -> Dict[str, npt.NDArray[np.float64]]
```

A função `sanitize_bounds(...)` valida e normaliza os limites das características (features).

**Parâmetros**:

* **bounds** (`Any`): Os limites de entrada, que devem ser `None` ou um dicionário com as chaves `'low'` e `'high'`.
* **problem_size** (`int`): O tamanho esperado para as listas de limites normalizadas, correspondente ao número de features do problema.

**Retorna**:

* `Dict[str, list]`: Dicionário no formato `{'low': [low_1, ..., low_N], 'high': [high_1, ..., high_N]}`.

## Distance

Funções utilitárias para distância normalizada entre matrizes com decoradores numba.

### Função `hamming(...)`

```python
@njit([(types.boolean[:], types.boolean[:])], cache=True)
def hamming(u: npt.NDArray[np.bool_], v: npt.NDArray[np.bool_]) -> float64:
```

Função para calcular a distância de Hamming normalizada entre dois pontos.

$((x₁ ≠ x₂) + (y₁ ≠ y₂) + ... + (yn ≠ yn)) / n$

**Parâmetros:**

* **u** (``npt.NDArray[np.bool_]``): Coordenadas do primeiro ponto
* **v** (``npt.NDArray[np.bool_]``): Coordenadas do segundo ponto.

**Returns:**

* Distância (``float64``) entre os dois pontos.

---

### Função `euclidean(...)`

```python
@njit()
def euclidean(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float64:
```

Função para calcular a distância euclidiana normalizada entre dois pontos.

$√( (x₁ - x₂)² + (y₁ - y₂)² + ... + (yn - yn)²)$

**Parâmetros:**

* **u** (``npt.NDArray[np.float64]``): Coordenadas do primeiro ponto
* **v** (``npt.NDArray[np.float64]``): Coordenadas do segundo ponto.

**Returns:**

* Distância (``float64``) entre os dois pontos.

---

### Função `cityblock(...)`

```python
@njit()
def cityblock(u: npt.NDArray[float64], v: npt.NDArray[float64]) -> float64:
```

Função para calcular a distância Manhattan normalizada entre dois pontos.

$(|x₁ - x₂| + |y₁ - y₂| + ... + |yn - yn|) / n$

**Parâmetros:**

* **u** (``npt.NDArray[float64]``): Coordenadas do primeiro ponto
* **v** (``npt.NDArray[float64]``): Coordenadas do segundo ponto.

**Returns:**

* Distância (``float64``) entre os dois pontos.

---

### Função `minkowski(...)`

```python
@njit()
def minkowski(
    u: npt.NDArray[float64],
    v: npt.NDArray[float64],
    p: float = 2.0
) -> float64:
```

Função para calcular a distância de Minkowski normalizada entre dois pontos.

$(( |X₁ - Y₁|p + |X₂ - Y₂|p + ... + |Xn - Yn|p) ¹/ₚ) / n$

**Parâmetros:**

* **u** (``npt.NDArray[float64]``): Coordenadas do primeiro ponto.
* **v** (``npt.NDArray[float64]``): Coordenadas do segundo ponto.
* **p** (``float``, padrão=2.0): O parâmetro p define o tipo de distância a ser calculada:
  * p = 1: Distância **Manhattan** - soma das diferenças absolutas.
  * p = 2: Distância **Euclidiana** - soma das diferenças ao quadrado (raiz quadrada).
  * p > 2: Distância **Minkowski** com uma penalidade crescente à medida que p aumenta.

**Returns:**

* Distância (``float``) entre os dois pontos.

---

### Função `compute_metric_distance(...)`

```python
@njit([(types.float64[:], types.float64[:], types.int32, types.float64)], cache=True)
def compute_metric_distance(
    u: npt.NDArray[float64],
    v: npt.NDArray[float64],
    metric: int,
    p: float = 2.0
) -> float64:
```

Função para calcular a distância entre dois pontos pela ``métrica`` escolhida.

**Parâmetros:**

* **u** (``npt.NDArray[float64]``): Coordenadas do primeiro ponto.
* **v** (``npt.NDArray[float64]``): Coordenadas do segundo ponto.
* **metric** (``int``): Métrica de distância a ser utilizada. Opções disponíveis: 0 (Euclidean), 1 (Manhattan), 2 (Minkowski).
* **p** (``float``, padrão=2.0): Parâmetro da métrica de Minkowski (utilizado apenas se `metric` for "minkowski").

**Returns:**

* Distância (``double``) entre os dois pontos com a métrica selecionada.

---

### Função `min_distance_to_class_vectors(...)`

```python
@njit([(types.float64[:, :], types.float64[:], types.int32, types.float64)], cache=True)
def min_distance_to_class_vectors(
    x_class: npt.NDArray[float64],
    vector_x: npt.NDArray[float64],
    metric: int,
    p: float = 2.0,
) -> float:
```

Calcula a menor distância entre um vetor de entrada e os vetores de uma classe.

**Parâmetros:**

* **x_class** (``npt.NDArray``): Array contendo os vetores da classe com os quais o vetor de entrada será comparado. Formato esperado: (n_amostras, n_características).
* **vector_x** (``npt.NDArray``): Vetor a ser comparado com os vetores da classe. Formato esperado: (n_características,).
* **metric** (``int``): Métrica de distância a ser utilizada. Opções disponíveis: 0 (Euclidean), 1 (Manhattan), 2 (Minkowski), 3 (Hamming).
* **p** (``float``): Parâmetro da métrica de Minkowski (utilizado apenas se `metric` for "minkowski").

**Returns:**

* float: A menor distância calculada entre o vetor de entrada e os vetores da classe.
* Retorna -1.0 se as dimensões de entrada forem incompatíveis.

---

### Função `get_metric_code(...)`

```python
def get_metric_code(metric: str) -> int:
```

Retorna o código numérico associado a uma métrica de distância.
  
**Parâmetros:**

* **metric** (``str``): Nome da métrica. Pode ser "euclidean", "manhattan", "minkowski" ou "hamming".

**Raises**

* ``ValueError``: Se a métrica informada não for suportada.

**Returns:**

* ``int``: Código numérico correspondente à métrica.

---

## Validation

### Função `detect_vector_data_type(...)`

```python
def detect_vector_data_type(
    vector: npt.NDArray
) -> FeatureType:
```

Detecta o tipo de dado em um determinado vetor.

Esta função analisa o vetor de entrada e classifica seus dados como um dos tipos suportados:

* **binário**: Valores booleanos (`True`/`False`) ou inteiro `0`/`1`.
* **contínuo**: Valores float dentro do intervalo normalizado `[0.0, 1.0]`.
* **intervalo**: Valores float fora do intervalo normalizado.

**Parâmetros:**

* **vector** (`npt.NDArray`): Um array contendo os dados a serem classificados.

**Retorna:**

* `FeatureType` (`Literal["binary-features", "continuous-features", "ranged-features"]`): O tipo de dado detectado no vetor.

**Gera:**

* `UnsupportedDataTypeError`: Gerado se o vetor contiver um tipo de dado não suportado.

---

### Função `check_array_type(...)`

```python
def check_array_type(x, name: str = "X") -> npt.NDArray:
```

Garante que o parâmetro recebido é um array numpy. Converte de lista se necessário.

**Parâmetros:**

* **x**: Array ou lista contendo as amostras e características.
* **name**: Nome da variável para mensagens de erro.

**Retorna:**

* `npt.NDArray`: O array convertido ou validado.

**Exceções:**

* `TypeError`: Se não for possível converter para ndarray.

---

### Função `check_shape_match(...)`

```python
def check_shape_match(x: npt.NDArray, y: npt.NDArray):
```

Garante que os arrays `x` e `y` possuem o mesmo número de amostras (primeira dimensão).

**Parâmetros:**

* **x**: Array de amostras.
* **y**: Array de classes alvo.

**Exceções:**

* `TypeError`: Se as dimensões não forem compatíveis.

---

### Função `check_feature_dimension(...)`

```python
def check_feature_dimension(x: npt.NDArray, expected: int):
```

Garante que o array possui o número esperado de características (features).

**Parâmetros:**

* **x**: Array de entrada para predição.
* **expected**: Número esperado de características por amostra.

**Exceções:**

* `FeatureDimensionMismatch`: Se o número de características não corresponder ao esperado.

---

### Função `check_binary_array(...)`

```python
def check_binary_array(x: npt.NDArray):
```

Garante que o array contém apenas valores 0 e 1.

**Parâmetros:**

* **x**: Array a ser verificado.

**Exceções:**

* `ValueError`: Se o array contiver valores diferentes de 0 e 1.

---

## Display

Funções utilitárias para exibir informações de algoritmos

### Função `_supports_box_drawing()`

```python
def _supports_box_drawing() -> bool
```

Função para verificar se o terminal suporta caracteres de borda.

**Retorna**:

* **bool** (`bool`): True se o terminal provavelmente suporta caracteres de borda, False caso contrário.

---

### class TableFormatter

Classe para formatar dados tabulares em strings para exibição no console.

**Parâmetros**:

* **headers** (`Mapping[str, int]`): Mapeamento dos nomes das colunas para suas larguras respectivas, no formato `{nome_coluna: largura_coluna}`.

**Exceções**:

* `ValueError`: Se `headers` estiver vazio ou não for um mapeamento válido.

---

#### Função `_border(left, middle, right, line, new_line=True)`

```python
def _border(self, left: str, middle: str, right: str, line: str, new_line: bool = True) -> str
```

Cria uma borda horizontal para a tabela.

**Parâmetros**:

* **left** (`str`): Caractere na borda esquerda.
* **middle** (`str`): Caractere separador entre colunas.
* **right** (`str`): Caractere na borda direita.
* **line** (`str`): Caractere usado para preencher a borda.
* **new_line** (`bool`, opcional): Se True, adiciona uma quebra de linha antes da borda (padrão é True).

**Retorna**:

* **border** (`str`): String representando a borda horizontal.

---

#### Função `get_header()`

```python
def get_header(self) -> str
```

Gera o cabeçalho da tabela, incluindo a borda superior, os títulos das colunas e a linha separadora.

**Retorna**:

* **header** (`str`): String formatada do cabeçalho da tabela.

---

#### Função `get_row(values)`

```python
def get_row(self, values: Mapping[str, Union[str, int, float]]) -> str
```

Gera uma linha formatada para os dados da tabela.

**Parâmetros**:

* **values** (`Mapping[str, Union[str, int, float]]`): Dicionário com os valores de cada coluna, no formato `{nome_coluna: valor}`.

**Retorna**:

* **row** (`str`): String formatada da linha da tabela.

---

#### Função `get_bottom(new_line=False)`

```python
def get_bottom(self, new_line: bool = False) -> str
```

Gera a borda inferior da tabela.

**Parâmetros**:

* **new_line** (`bool`, opcional): Se True, adiciona uma quebra de linha antes da borda (padrão é False).

**Retorna**:

* **bottom** (`str`): String formatada da borda inferior.

---

### class ProgressTable(TableFormatter)

Classe para exibir uma tabela formatada no console para acompanhar o progresso de um algoritmo.

**Parâmetros**:

* **headers** (`Mapping[str, int]`): Mapeamento `{nome_coluna: largura_coluna}`.
* **verbose** (`bool`, padrão=True): Se False, não imprime nada no terminal.

**Exceções**:

* `ValueError`: Se `headers` estiver vazio ou não for um mapeamento válido.

---

#### Função `_print_header()`

```python
def _print_header(self) -> None
```

Imprime o cabeçalho da tabela.

---

#### Função `update(values)`

```python
def update(self, values: Mapping[str, Union[str, int, float]]) -> None
```

Adiciona uma nova linha de valores à tabela.

**Parâmetros**:

* **values** (`Mapping[str, Union[str, int, float]]`): As chaves devem corresponder às colunas definidas em `headers`.

---

#### Função `finish()`

```python
def finish(self) -> None
```

Encerra a exibição da tabela, imprimindo a borda inferior e o tempo total.

---
