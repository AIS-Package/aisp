# Utils

Funções de utilidade para o desenvolvimento.

## Metrics

### def accuracy_score(...)

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

## Multiclass

### def predict_knn_affinity(...)

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

* **_X_** (`npt.NDArray`): Dados de entrada a serem classificados.
* **_k_** (`int`): Número de vizinhos mais próximos a considerar para a previsão.
* **_all_cell_vectors_** (`List[Tuple[Union[str, int], npt.NDArray]]`): Lista de tuplas contendo pares (nome_da_classe, vetor_da_célula).
* **_affinity_func_** (`Callable[[npt.NDArray, npt.NDArray], float]`): Função que recebe dois vetores e retorna um valor de afinidade.

**Retorna:**

* `npt.NDArray`: Array de rótulos previstos para cada amostra em X, baseado nos k vizinhos mais próximos.

---

### def slice_index_list_by_class(...)

```python
def slice_index_list_by_class(classes: Optional[Union[npt.NDArray, list]], y: npt.NDArray) -> dict
```

A função ``slice_index_list_by_class(...)``, separa os índices das amostras
conforme a classe de saída, para percorrer o array de amostra apenas nas posições
onde a saída corresponde à classe sendo treinada.

**Parâmetros:**

* **_classes_** (`Optional[Union[npt.NDArray, list]]`): Lista com classes únicas. Se None, retorna um dicionário vazio.
* **_y_** (`npt.NDArray`): Array com as classes de saída do array de amostra ``X``.

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

### def sanitize_choice(...)

```python
def sanitize_choice(value: T, valid_choices: Iterable[T], default: T) -> T
```

A função ``sanitize_choice(...)``, retorna o valor se estiver presente no conjunto de opções válidas; caso contrário, retorna o valor padrão.

**Parameters:**

* _**value**_ (``T``): O valor a ser verificado.
* _**valid_choices**_ (``Iterable[T]``): Uma coleção de opções válidas.
* _**default**_: O valor padrão a ser retornado se ``value`` não estiver em ``valid_choices``.

**Returns:**

* `T`: O valor original, se válido, ou o valor padrão, se não.

---

### def sanitize_param(...)

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

### def sanitize_seed(...)

```python
def sanitize_seed(seed: Any) -> Optional[int]:
```

A função ``sanitize_param(...)``, retorna a semente se for um inteiro não negativo; caso contrário, retorna Nenhum.

**Parameters:**

* seed (``Any``): O valor da seed a ser validado.

**Returns:**

* ``Optional[int]``: A seed original se for um inteiro não negativo, ou ``None`` se for inválido.

---

### def sanitize_bounds(...)

```python
def sanitize_bounds(bounds: Any, problem_size: int) -> Dict[str, npt.NDArray[np.float64]]
```

A função `sanitize_bounds(...)` valida e normaliza os limites das características (features).

**Parâmetros**:

* _**bounds**_ (`Any`): Os limites de entrada, que devem ser `None` ou um dicionário com as chaves `'low'` e `'high'`.
* _**problem_size**_ (`int`): O tamanho esperado para as listas de limites normalizadas, correspondente ao número de features do problema.

**Retorna**:

* `Dict[str, list]`: Dicionário no formato `{'low': [low_1, ..., low_N], 'high': [high_1, ..., high_N]}`.

## Distance

Funções utilitárias para distância normalizada entre matrizes com decoradores numba.

### def hamming(...)

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

### def euclidean(...)

```python
def euclidean(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> np.float64:
```

Função para calcular a distância euclidiana normalizada entre dois pontos.

$√( (x₁ - x₂)² + (y₁ - y₂)² + ... + (yn - yn)²)$

**Parameters:**

* u (``npt.NDArray``): Coordenadas do primeiro ponto
* v (``npt.NDArray``): Coordenadas do segundo ponto.

**Returns:**

* Distância (``float``) entre os dois pontos.

---

### def cityblock(...)

```python
def cityblock(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> np.float64:
```

Função para calcular a distância Manhattan normalizada entre dois pontos.

$(|x₁ - x₂| + |y₁ - y₂| + ... + |yn - yn|) / n$

**Parameters:**

* u (``npt.NDArray``): Coordenadas do primeiro ponto
* v (``npt.NDArray``): Coordenadas do segundo ponto.

**Returns:**

* Distância (``float``) entre os dois pontos.

---

### def minkowski(...)

```python
def minkowski(u: npt.NDArray[np.float64], v: npt.NDArray[np.float64], p: float = 2.0):
```

Função para calcular a distância de Minkowski normalizada entre dois pontos.

$(( |X₁ - Y₁|p + |X₂ - Y₂|p + ... + |Xn - Yn|p) ¹/ₚ) / n$

**Parameters:**

* u (``npt.NDArray``): Coordenadas do primeiro ponto.
* v (``npt.NDArray``): Coordenadas do segundo ponto.
* p (``float``): O parâmetro p define o tipo de distância a ser calculada:
  * p = 1: Distância **Manhattan** — soma das diferenças absolutas.
  * p = 2: Distância **Euclidiana** — soma das diferenças ao quadrado (raiz quadrada).
  * p > 2: Distância **Minkowski** com uma penalidade crescente à medida que p aumenta.

**Returns:**

* Distância (``float``) entre os dois pontos.

---

### def compute_metric_distance(...)

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

### def min_distance_to_class_vectors(...)

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

### def get_metric_code(...)

```python
def get_metric_code(metric: str) -> int:
```

Retorna o código numérico associado a uma métrica de distância.
  
**Parameters:**

* metric (``str``): Nome da métrica. Pode ser "euclidean", "manhattan", "minkowski" ou "hamming".

**Raises**

* ``ValueError``: Se a métrica informada não for suportada.

**Returns:**

* ``int``: Código numérico correspondente à métrica.

---

## Validation

### def detect_vector_data_type(...)

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

#### Parâmetros

* `vetor` (`npt.NDArray`): Um array contendo os dados a serem classificados.

#### Retorna

* `FeatureType` (`Literal["binary-features", "continuous-features", "ranged-features"]`): O tipo de dado detectado no vetor.

#### Gera

* `UnsupportedDataTypeError`: Gerado se o vetor contiver um tipo de dado não suportado.

---

### def check_array_type(...)

```python
def check_array_type(x, name: str = "X") -> npt.NDArray:
```

Garante que o parâmetro recebido é um array numpy. Converte de lista se necessário.

**Parâmetros:**

* `x`: Array ou lista contendo as amostras e características.
* `name`: Nome da variável para mensagens de erro.

**Retorna:**

* `npt.NDArray`: O array convertido ou validado.

**Lança:**

* `TypeError`: Se não for possível converter para ndarray.

---

### def check_shape_match(...)

```python
def check_shape_match(x: npt.NDArray, y: npt.NDArray):
```

Garante que os arrays `x` e `y` possuem o mesmo número de amostras (primeira dimensão).

**Parâmetros:**

* `x`: Array de amostras.
* `y`: Array de classes alvo.

**Lança:**

* `TypeError`: Se as dimensões não forem compatíveis.

---

### def check_feature_dimension(...)

```python
def check_feature_dimension(x: npt.NDArray, expected: int):
```

Garante que o array possui o número esperado de características (features).

**Parâmetros:**

* `x`: Array de entrada para predição.
* `expected`: Número esperado de características por amostra.

**Lança:**

* `FeatureDimensionMismatch`: Se o número de características não corresponder ao esperado.

---

### def check_binary_array(...)

```python
def check_binary_array(x: npt.NDArray):
```

Garante que o array contém apenas valores 0 e 1.

**Parâmetros:**

* `x`: Array a ser verificado.

**Lança:**

* `ValueError`: Se o array contiver valores diferentes de 0 e 1.

---

## Display

Funções utilitárias para exibir informações de algoritmos

### def _supports_box_drawing()

```python
def _supports_box_drawing() -> bool
```

Função para verificar se o terminal suporta caracteres de borda.

**Retorna**:

* _**bool**_ (`bool`): True se o terminal provavelmente suporta caracteres de borda, False caso contrário.

---

### class TableFormatter

Classe para formatar dados tabulares em strings para exibição no console.

**Parâmetros**:

* _**headers**_ (`Mapping[str, int]`): Mapeamento dos nomes das colunas para suas larguras respectivas, no formato `{nome_coluna: largura_coluna}`.

**Exceções**:

* `ValueError`: Se `headers` estiver vazio ou não for um mapeamento válido.

---

#### def _border(left, middle, right, line, new_line=True)

```python
def _border(self, left: str, middle: str, right: str, line: str, new_line: bool = True) -> str
```

Cria uma borda horizontal para a tabela.

**Parâmetros**:

* _**left**_ (`str`): Caractere na borda esquerda.
* _**middle**_ (`str`): Caractere separador entre colunas.
* _**right**_ (`str`): Caractere na borda direita.
* _**line**_ (`str`): Caractere usado para preencher a borda.
* _**new_line**_ (`bool`, opcional): Se True, adiciona uma quebra de linha antes da borda (padrão é True).

**Retorna**:

* _**border**_ (`str`): String representando a borda horizontal.

---

#### def get_header()

```python
def get_header(self) -> str
```

Gera o cabeçalho da tabela, incluindo a borda superior, os títulos das colunas e a linha separadora.

**Retorna**:

* _**header**_ (`str`): String formatada do cabeçalho da tabela.

---

#### def get_row(values)

```python
def get_row(self, values: Mapping[str, Union[str, int, float]]) -> str
```

Gera uma linha formatada para os dados da tabela.

**Parâmetros**:

* _**values**_ (`Mapping[str, Union[str, int, float]]`): Dicionário com os valores de cada coluna, no formato `{nome_coluna: valor}`.

**Retorna**:

* _**row**_ (`str`): String formatada da linha da tabela.

---

#### def get_bottom(new_line=False)

```python
def get_bottom(self, new_line: bool = False) -> str
```

Gera a borda inferior da tabela.

**Parâmetros**:

* _**new_line**_ (`bool`, opcional): Se True, adiciona uma quebra de linha antes da borda (padrão é False).

**Retorna**:

* _**bottom**_ (`str`): String formatada da borda inferior.

---

### class ProgressTable(TableFormatter)

Classe para exibir uma tabela formatada no console para acompanhar o progresso de um algoritmo.

**Parâmetros**:

* _**headers**_ (`Mapping[str, int]`): Mapeamento `{nome_coluna: largura_coluna}`.
* _**verbose**_ (`bool`, padrão=True): Se False, não imprime nada no terminal.

**Exceções**:

* `ValueError`: Se `headers` estiver vazio ou não for um mapeamento válido.

---

#### def _print_header()

```python
def _print_header(self) -> None
```

Imprime o cabeçalho da tabela.

---

#### def update(values)

```python
def update(self, values: Mapping[str, Union[str, int, float]]) -> None
```

Adiciona uma nova linha de valores à tabela.

**Parâmetros**:

* _**values**_ (`Mapping[str, Union[str, int, float]]`): As chaves devem corresponder às colunas definidas em `headers`.

---

#### def finish()

```python
def finish(self) -> None
```

Encerra a exibição da tabela, imprimindo a borda inferior e o tempo total.

---
