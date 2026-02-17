# AiNet (Artificial Immune Network)

Esta classe estende a classe [**Base**](../../advanced-guides/base/clusterer.md).

## Construtor AiNet

A classe `AiNet` implementa o algoritmo de Rede Imune Artificial para **compressão** e **clustering**.
Ela utiliza princípios da teoria de redes imunes, seleção clonal e maturação por afinidade para comprimir conjuntos de dados e encontrar clusters.

Para clustering, pode opcionalmente utilizar uma **Árvore Geradora Mínima (MST)** para separar nós distantes em grupos.

**Atributos:**

* **N** (`int`): Número de células de memória (anticorpos) na população. Padrão: 50.
* **n_clone** (`int`): Número de clones gerados por célula de memória selecionada. Padrão: 10.
* **top_clonal_memory_size** (`Optional[int]`): Número de anticorpos de maior afinidade selecionados para clonagem. Padrão: 5.
* **n_diversity_injection** (`int`): Número de novos anticorpos aleatórios injetados para manter a diversidade. Padrão: 5.
* **affinity_threshold** (`float`): Limite para seleção/supressão de células. Padrão: 0.5.
* **suppression_threshold** (`float`): Limite para remoção de células de memória semelhantes. Padrão: 0.5
* **mst_inconsistency_factor** (`float`): Fator para determinar arestas inconsistentes na MST. Padrão: 2.0.
* **max_iterations** (`int`): Número máximo de iterações de treinamento. Padrão: 10.
* **k** (`int`): Número de vizinhos mais próximos usados para predição de rótulos. Padrão: 3.
* **metric** (Literal["manhattan", "minkowski", "euclidean"]): Forma de calcular a distância entre o detector e a amostra:

  * `'euclidean'` ➜ Distância dada pela expressão:
    √( (x₁ - x₂)² + (y₁ - y₂)² + ... + (yn - yn)²).
  * `'minkowski'` ➜ Distância dada pela expressão:
    ( |X₁ - Y₁|ᵖ + |X₂ - Y₂|ᵖ + ... + |Xn - Yn|ᵖ )^(¹/ₚ).
  * `'manhattan'` ➜ Distância dada pela expressão:
    ( |x₁ - x₂| + |y₁ - y₂| + ... + |yn - yn|).
    Padrão: "euclidean".

* **seed** (`Optional[int]`): Semente para geração de números aleatórios. Padrão: None.
* **use_mst_clustering** (`bool`): Define se o clustering baseado em MST deve ser utilizado. Padrão: True.
* **kwargs**:

  * **p** (`float`): Parâmetro para distância de Minkowski. Padrão: 2.

**Outras variáveis inicializadas:**

* **_population_antibodies** (``Optional[npt.NDArray]``): Conjunto atual de anticorpos.
* **_memory_network** (``Dict[int, List[Cell]]``): Dicionário que mapeia clusters para anticorpos.
* **_mst_structure** (``Optional[npt.NDArray]``): Estrutura de adjacência da MST.
* **_mst_mean_distance** (``Optional[float]``): Média das distâncias das arestas da MST.
* **_mst_std_distance** (``Optional[float]``): Desvio padrão das distâncias das arestas da MST.
* **labels** (``Optional[npt.NDArray]``): Lista de rótulos dos clusters.

---

## Métodos Públicos

### Método `fit(...)`

Treina o modelo AiNet com os dados de entrada:

```python
def fit(self, X: Union[npt.NDArray, list], verbose: bool = True) -> AiNet:
```

**Parâmetros:**

* **X** (`Union[npt.NDArray, list]`): Matriz com amostras (linhas) e atributos (colunas).
* **verbose** (`bool`): Booleano, padrão True, habilita feedback de progresso.

**Exceções**

* `TypeError`: Se X não for um ndarray ou uma lista.
* `UnsupportedTypeError`: Se o tipo de dados do vetor não for suportado.

*Retorna a instância da classe.*

---

### Método `predict(...)`

Prediz os rótulos dos clusters para novas amostras:

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
```

**Parâmetros:**

* **X** (`Union[npt.NDArray, list]`): Matriz de atributos de entrada.

**Exceções**

* `TypeError`: Se X não for um ndarray ou uma lista.
* `ValueError`: X contém valores que não são compostos apenas por 0 e 1.
* `FeatureDimensionMismatch`: Se o número de características em X não corresponder ao número esperado.
* `ModelNotFittedError`: Se o modelo ainda não tiver sido ajustado e não possuir células de memória definidas, não conseguirá realizar predições.

**Retorna:**

* **Predictions**: Matriz de rótulos de clusters, ou None caso o clustering esteja desabilitado.

---

### Método `update_clusters(...)`

Particiona os clusters utilizando a MST:

```python
def update_clusters(self, mst_inconsistency_factor: Optional[float] = None):
```

**Parâmetros:**

* **mst_inconsistency_factor** (`Optional[float]`): Valor opcional (float) para sobrescrever o fator de inconsistência da MST.

**Atualiza:**

* **_memory_network**: Dicionário de rótulos de clusters para vetores de anticorpos.
* **labels**: Lista de rótulos de clusters.

---

## Métodos Privados

### Método `_init_population_antibodies(...)`

Inicializa a população de anticorpos aleatoriamente.

```python
def _init_population_antibodies(self) -> npt.NDArray:
```

**Retorna:** Anticorpos inicializados (`npt.NDArray`).

---

### Método` _select_and_clone_population(...)`

Seleciona os melhores anticorpos e gera clones mutados:

```python
def _select_and_clone_population(self, antigen: npt.NDArray, population: npt.NDArray) -> list:
```

**Parâmetros:**

* **antigen** (`npt.NDArray`): Vetor representando o antígeno para o qual as afinidades serão calculadas.
* **population** (`npt.NDArray`): Matriz de anticorpos a serem avaliados e clonados.

**Retorna:** Lista de clones mutados.

---

### Método `_clonal_suppression(...)`

Suprime clones redundantes com base em limiares:

```python
def _clonal_suppression(self, antigen: npt.NDArray, clones: list):
```

**Parâmetros:**

* **antigen** (`npt.NDArray`): Vetor representando o antígeno.
* **clones** (`list`): Lista de clones candidatos a serem suprimidos.

**Retorna:** Lista de clones não redundantes e de alta afinidade.

---

### Método `_memory_suppression(...)`

Remove anticorpos redundantes da memória:

```python
def _memory_suppression(self, pool_memory: list) -> list:
```

**Parâmetros:**

* **pool_memory** (`list`): Lista de anticorpos atualmente na memória.

**Retorna:**

Memória filtrada (`list`).

---

### Método `_diversity_introduction(...)`

Introduz novos anticorpos para manter a diversidade.

```python
def _diversity_introduction(self) -> npt.NDArray:
```

**Retorna:** Conjunto de novos anticorpos (`npt.NDArray`).

---

### Método `_affinity(...)`

Calcula o estímulo entre dois vetores:

```python
def _affinity(self, u: npt.NDArray, v: npt.NDArray) -> float:
```

**Parâmetros:**

* **u** (`npt.NDArray`): Vetor representando o primeiro ponto.
* **v** (`npt.NDArray`): Vetor representando o segundo ponto.

**Retorna:** Valor de afinidade (`float`) no intervalo [0,1].

---

### Método `_calculate_affinities(...)`

Calcula a matriz de afinidades entre um vetor de referência e vetores-alvo:

```python
def _calculate_affinities(self, u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
```

**Parâmetros:**

* **u** (`npt.NDArray`): Vetor de referência (`npt.NDArray`) de formato `(n_features,)`.
* **v** (`npt.NDArray`): Vetores-alvo (`npt.NDArray`) de formato `(n_samples, n_features)`.

**Retorna:** Vetor de afinidades (`npt.NDArray`) com formato `(n_samples,)`.

---

### Método `_clone_and_mutate(...)`

Gera clones mutados:

```python
def _clone_and_mutate(self, antibody: npt.NDArray, n_clone: int) -> npt.NDArray:
```

**Parâmetros:**

* **antibody** (`npt.NDArray`): Vetor de anticorpo original a ser clonado e mutado.
* **n_clone** (`int`): Número de clones a serem gerados.

**Retorna:** Matriz de clones mutados (`npt.NDArray`) de formato `(n_clone, len(antibody))`.

---

### Método `_build_mst(...)`

Constrói a MST e armazena estatísticas:

```python
def _build_mst(self):
```

**Exceções:** ValueError se a população de anticorpos estiver vazia.

**Atualiza variáveis internas:**

* **_mst_structure**: Estrutura de adjacência da MST.
* **_mst_mean_distance**: Distância média das arestas.
* **_mst_std_distance**: Desvio padrão das distâncias das arestas da MST.

---

## Referências

> 1. De Castro, Leandro & José, Fernando & von Zuben, Antonio Augusto. (2001). aiNet: An Artificial Immune Network for Data Analysis.
> 2. Disponível em: [https://www.researchgate.net/publication/228378350_aiNet_An_Artificial_Immune_Network_for_Data_Analysis](https://www.researchgate.net/publication/228378350_aiNet_An_Artificial_Immune_Network_for_Data_Analysis)

> 2. SciPy Documentation. *Minimum Spanning Tree*.
>    Disponível em: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree)
