# AiNet (Artificial Immune Network)

Esta classe estende a classe [**Base**](../../advanced-guides/base/clusterer.md).

## Construtor AiNet:

A classe `AiNet` implementa o algoritmo de Rede Imune Artificial para **compressão** e **clustering**.
Ela utiliza princípios da teoria de redes imunes, seleção clonal e maturação por afinidade para comprimir conjuntos de dados e encontrar clusters.

Para clustering, pode opcionalmente utilizar uma **Árvore Geradora Mínima (MST)** para separar nós distantes em grupos.

**Atributos:**

* **N** (`int`): Número de células de memória (anticorpos) na população. Padrão: 50.

* **n_clone** (`int`): Número de clones gerados por célula de memória selecionada. Padrão: 10.

* **top_clonal_memory_size** (`Optional[int]`): Número de anticorpos de maior afinidade selecionados para clonagem. Padrão: 5.

* **n_diversity_injection** (`int`): Número de novos anticorpos aleatórios injetados para manter a diversidade. Padrão: 5.

* **affinity_threshold** (`float`): Limite para seleção/supressão de células. Padrão: 0.5.

* **suppression_threshold** (`float`): Limite para remoção de células de memória semelhantes. Padrão: 0.5.

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

* **_population_antibodies** (`npt.NDArray`): Conjunto atual de anticorpos.
* **_memory_network** (`dict`): Dicionário que mapeia clusters para anticorpos.
* **_mst_structure** (`scipy.sparse.csr_matrix`): Estrutura de adjacência da MST.
* **_mst_mean_distance** (`float`): Média das distâncias das arestas da MST.
* **_mst_std_distance** (`float`): Desvio padrão das distâncias das arestas da MST.
* **classes** (`list`): Lista de rótulos dos clusters.

---

## Métodos Públicos

### Função fit(...)

Treina o modelo AiNet com os dados de entrada:

```python
def fit(self, X: npt.NDArray, verbose: bool = True):
```

**Parâmetros de entrada:**

* **X**: Matriz com amostras (linhas) e atributos (colunas).
* **verbose**: Booleano, padrão True, habilita feedback de progresso.

*Retorna a instância da classe.*

---

### Função predict(...)

Prediz os rótulos dos clusters para novas amostras:

```python
def predict(self, X) -> Optional[npt.NDArray]:
```

**Parâmetros de entrada:**

* **X**: Matriz de atributos de entrada.

**Retorna:**

* **Predictions**: Matriz de rótulos de clusters, ou None caso o clustering esteja desabilitado.

---

### Função update_clusters(...)

Particiona os clusters utilizando a MST:

```python
def update_clusters(self, mst_inconsistency_factor: Optional[float] = None):
```

**Parâmetros de entrada:**

* **mst_inconsistency_factor**: Valor opcional (float) para sobrescrever o fator de inconsistência da MST.

**Atualiza:**

* **_memory_network**: Dicionário de rótulos de clusters para vetores de anticorpos.
* **classes**: Lista de rótulos de clusters.

---

## Métodos Privados

### Função _init_population_antibodies(...)

Inicializa a população de anticorpos aleatoriamente.

```python
def _init_population_antibodies(self) -> npt.NDArray:
```

**Parâmetros de entrada:** Nenhum

**Retorna:** Anticorpos inicializados (`npt.NDArray`).

---

### Função _select_and_clone_population(...)

Seleciona os melhores anticorpos e gera clones mutados:

```python
def _select_and_clone_population(self, antigen: npt.NDArray, population: npt.NDArray) -> list:
```

**Parâmetros de entrada:**

* **antigen**: Vetor representando o antígeno para o qual as afinidades serão calculadas.
* **population**: Matriz de anticorpos a serem avaliados e clonados.

**Retorna:** Lista de clones mutados.

---

### Função _clonal_suppression(...)

Suprime clones redundantes com base em limiares:

```python
def _clonal_suppression(self, antigen: npt.NDArray, clones: list):
```

**Parâmetros de entrada:**

* **antigen**: Vetor representando o antígeno.
* **clones**: Lista de clones candidatos a serem suprimidos.

**Retorna:** Lista de clones não redundantes e de alta afinidade.

---

### Função _memory_suppression(...)

Remove anticorpos redundantes da memória:

```python
def _memory_suppression(self, pool_memory: list) -> list:
```

**Parâmetros de entrada:**

* **pool_memory**: Lista de anticorpos atualmente na memória.

**Retorna:** Memória filtrada (`list`).

---

### Função _diversity_introduction(...)

Introduce novos anticorpos aleatórios:

```python
def _diversity_introduction(self) -> npt.NDArray:
```

**Parâmetros de entrada:** Nenhum

**Retorna:** Conjunto de novos anticorpos (`npt.NDArray`).

---

### Função _affinity(...)

Calcula o estímulo entre dois vetores:

```python
def _affinity(self, u: npt.NDArray, v: npt.NDArray) -> float:
```

**Parâmetros de entrada:**

* **u**: Vetor representando o primeiro ponto.
* **v**: Vetor representando o segundo ponto.

**Retorna:** Valor de afinidade (`float`) no intervalo [0,1].

---

### Função _calculate_affinities(...)

Calcula a matriz de afinidades entre um vetor de referência e vetores-alvo:

```python
def _calculate_affinities(self, u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
```

**Parâmetros de entrada:**

* **u**: Vetor de referência (`npt.NDArray`) de formato `(n_features,)`.
* **v**: Vetores-alvo (`npt.NDArray`) de formato `(n_samples, n_features)`.

**Retorna:** Vetor de afinidades (`npt.NDArray`) com formato `(n_samples,)`.

---

### Função _clone_and_mutate(...)

Gera clones mutados:

```python
def _clone_and_mutate(self, antibody: npt.NDArray, n_clone: int) -> npt.NDArray:
```

**Parâmetros de entrada:**

* **antibody**: Vetor de anticorpo original a ser clonado e mutado.
* **n_clone**: Número de clones a serem gerados.

**Retorna:** Matriz de clones mutados (`npt.NDArray`) de formato `(n_clone, len(antibody))`.

---

### Função _build_mst(...)

Constrói a MST e armazena estatísticas:

```python
def _build_mst(self):
```

**Parâmetros de entrada:** Nenhum

**Exceções:** ValueError se a população de anticorpos estiver vazia.

**Atualiza variáveis internas:**

* **_mst_structure**: Estrutura de adjacência da MST.
* **_mst_mean_distance**: Distância média das arestas.
* **_mst_std_distance**: Desvio padrão das distâncias das arestas da MST.

---

# Referências

> 1. De Castro, Leandro & José, Fernando & von Zuben, Antonio Augusto. (2001). aiNet: An Artificial Immune Network for Data Analysis.
> 2. Disponível em: [https://www.researchgate.net/publication/228378350_aiNet_An_Artificial_Immune_Network_for_Data_Analysis](https://www.researchgate.net/publication/228378350_aiNet_An_Artificial_Immune_Network_for_Data_Analysis)

> 2. SciPy Documentation. *Minimum Spanning Tree*.
>    Disponível em: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.minimum_spanning_tree)
