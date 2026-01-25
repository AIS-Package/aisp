# Algoritmo de Seleção Clonal (CLONALG)

## Clonalg

A classe `Clonalg` é um **algoritmo de otimização** inspirado no processo biológico de seleção clonal do sistema
imunológico. Esta implementação é projetada para minimizar ou maximizar funções de custo em diversos tipos de
problemas, incluindo problemas binários, contínuos, com valores limitados (ranged) e de permutação.

A implementação do CLONALG foi inspirada na descrição apresentada em [1](#ref1).

Esta implementação do CLONALG contém algumas alterações baseadas no contexto do AISP, para aplicação geral
a diversos problemas, que podem produzir resultados diferentes da implementação padrão ou
específica. Esta adaptação visa generalizar o CLONALG para tarefas de minimização e
maximização, além de suportar problemas contínuos, discretos e de permutação.

---

### Construtor

O construtor inicializa a instância do CLONALG com os principais parâmetros que definem o processo de otimização.

**Atributos:**

* **problem_size** (`int`): Dimensão do problema a ser otimizado.
* **N** (`int`, padrão=50): Número de células de memória (anticorpos) na população.
* **rate_clonal** (`float`, padrão=10): Número máximo de clones possíveis de uma célula. Este valor é multiplicado pela afinidade da célula para determinar o número de clones.
* **rate_hypermutation** (`float`, padrão=1.0): Taxa de clones mutados, usada como fator escalar.
* **n_diversity_injection** (`int`, padrão=5): Número de novas células de memória aleatórias injetadas para manter a diversidade.
* **selection_size** (`int`, padrão=5): Número de melhores anticorpos selecionados para clonagem.
* **affinity_function** (`Optional[Callable[..., npt.NDArray]]`, padrão=None): Função objetivo usada para avaliar soluções candidatas.
* **feature_type** (`FeatureTypeAll`, padrão='ranged-features'): Tipo de amostra do problema, podendo ser `'continuous-features'`, `'binary-features'`, `'ranged-features'` ou `'permutation-features'`.
* **bounds** (`Optional[Dict]`, padrão=None): Dicionário definindo os limites de busca para problemas `'ranged-features'`. Pode ser um único intervalo ou uma lista de intervalos para cada dimensão.
* **mode** (`Literal["min", "max"]`, padrão="min"): Especifica se o algoritmo minimiza ou maximiza a função de custo.
* **seed** (`Optional[int]`, padrão=None): Semente para geração de números aleatórios.

---

## Métodos Públicos

### Função `optimize(...)`

```python
def optimize(
    self,
    max_iters: int = 50,
    n_iter_no_change=10,
    verbose: bool = True
) -> List[Antibody]:
```

Este método executa o processo de otimização e retorna a população de anticorpos.

**Parâmetros:**

* **max_iters** (`int`, padrão=50): Número máximo de interações.
* **n_iter_no_change** (`int`, padrão=10): Número máximo de iterações sem melhoria na melhor solução.
* **verbose** (`bool`, padrão=True): Flag para habilitar ou desabilitar saída detalhada durante o processo de otimização.

**Retorna:**

* `npt.NDArray`: A população de anticorpos após a expansão clonal.

---

#### Função `affinity_function(...)`

```python
def affinity_function(self, solution: npt.NDArray) -> np.float64:
```

Este método avalia a afinidade de uma solução candidata. Levanta `NotImplementedError` se nenhuma função de afinidade tiver sido fornecida à instância da classe.

**Parâmetros:**

* **solution** (`npt.NDArray`): Solução candidata a ser avaliada.

**Retorna:**

* `np.float64`: Valor de afinidade associado à solução.

---

### Métodos Privados

#### Função `_select_top_antibodies(...)`

```python
def _select_top_antibodies(self, n: int, antibodies: list[tuple]) -> list[tuple]:
```

Seleciona os `n` melhores anticorpos com base em suas pontuações de afinidade, de acordo com o `mode` (`'min'` ou `'max'`).

**Parâmetros:**

* **n** (`int`): Número de anticorpos a serem selecionados.
* **antibodies* (`list[tuple]`): Lista de tuplas, onde cada tupla representa um anticorpo e sua pontuação associada.

**Retorna:**

* `list[tuple]`: Lista contendo os `n` anticorpos selecionados.

---

#### Função `_init_population_antibodies(...)`

```python
def _init_population_antibodies(self) -> npt.NDArray:
```

Inicializa aleatoriamente a população inicial de anticorpos.

**Retorna:**

* `npt.NDArray`: Lista com os anticorpos inicializados.

---

#### Função `_diversity_introduction(...)`

```python
def _diversity_introduction(self):
```

Introduz novos anticorpos aleatórios na população para manter a diversidade genética e ajudar a evitar convergência prematura.

**Retorna:**

* `npt.NDArray`: Array contendo os novos anticorpos aleatórios.

---

#### Função `_clone_and_mutate(...)`

```python
def _clone_and_mutate(self, antibody: npt.NDArray, n_clone: int, rate_hypermutation: float) -> npt.NDArray:
```

Gera clones mutados a partir de um único anticorpo. A estratégia de mutação depende do `feature_type` especificado durante a inicialização (`'binary-features'`, `'continuous-features'`, `'ranged-features'` ou `'permutation-features'`).

**Parâmetros:**

* **antibody** (`npt.NDArray`): Vetor original do anticorpo a ser clonado e mutado.
* **n_clone** (`int`): Número de clones a serem gerados.
* **rate_hypermutation** (`float`): Taxa de hipermutação.

**Retorna:**

* `npt.NDArray`: Array contendo os clones mutados.

---

#### Função `_clone_and_hypermutation(...)`

```python
def _clone_and_hypermutation(
    self,
    population: list[Antibody]
) -> list[Antibody]:
```

Clona e aplica hipermutação a uma população de anticorpos. Retorna uma lista de todos os clones e suas afinidades em relação à função de custo.

**Parâmetros:**

* **population** (`list[tuple]`): Lista de anticorpos a serem avaliados e clonados.

**Retorna:**

* `list[Antibody]`: Lista contendo os clones mutados.

---

## Referências

<br id='ref1'/>

> 1. BROWNLEE, Jason. Clonal Selection Algorithm. Clever Algorithms: Nature-inspired Programming Recipes., 2011.
> Available at: <https://cleveralgorithms.com/nature-inspired/immune/clonal_selection_algorithm.html>
