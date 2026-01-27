
# BNSA (Algoritmo de Seleção Negativa Binária)

Esta classe estende a classe [**Base**](../../advanced-guides/base/classifier.md).

## Construtor BNSA

A classe ``BNSA`` tem a finalidade de classificação e identificação de anomalias através do método self e not self.

**Attributes:**

* **N** (``int``): Quantidade de detectores. Defaults to ``100``.
* **aff_thresh** (``float``): A variável representa a porcentagem de não similaridade entre a célula T e as amostras próprias. O valor padrão é de 10% (0.1), enquanto que o valor de 1.0 representa 100% de não similaridade.
* **max_discards** (``int``): Este parâmetro indica o número máximo de descartes de detectores em sequência, cujo objetivo é evitar um
possível loop infinito caso seja definido um raio que não, seja possível gerar detectores do não-próprio. Defaults to ``100``.
* **seed** (``int``): Semente para a geração randômica dos valores nos detectores. Defaults to ``None``.
* no_label_sample_selection (``str``): Método para a seleção de rótulos para amostras designadas como não pertencentes por todos os detectores não pertencentes. **Tipos de métodos disponíveis:**
  * (``max_average_difference``): Seleciona a classe com a maior diferença média entre os detectores.
  * (``max_nearest_difference``): Seleciona a classe com a maior diferença entre o detector mais próximo e mais distante da amostra.

**Outras variáveis iniciadas:**

* **detectors** (``dict``): Esta variável armazena uma lista de detectores por classe.
* **classes** (``npt.NDArray``): lista de classes de saída.

### Método `fit(...)`

A função ``fit(...)`` gera os detectores para os não próprios com relação às amostras:

```python
def fit(
    self,
    X: Union[npt.NDArray, list],
    y: Union[npt.NDArray, list],
    verbose: bool = True,
) -> BNSA:
```

Nela é realizado o treinamento de acordo com ``X`` e ``y``, usando o método de seleção negativa(``NegativeSelect``).

**Os parâmetros de entrada são:**

* **X** (`Union[npt.NDArray, list]`): array com as características das amostras com **N** amostras (linhas) e **N** características  (colunas), normalizados para valores entre [0, 1].
* **y** (`Union[npt.NDArray, list]`): array com as classes de saídas disposto em **N** amostras que são relacionadas ao ``X``.
* **verbose** (`bool`): boolean com valor default ``True``, determina se o feedback da geração dos detectores será imprimido.

**Lança:**

* ``TypeError``: Se X ou y não forem ndarrays, ou tiverem formas incompatíveis.
* ``ValueError``: X contém valores que não são compostos apenas por 0 e 1.
* ``MaxDiscardsReachedError``: O número máximo de descartes do detector foi atingido durante a maturação. Verifique o valor do raio definido e considere reduzi-lo.

*Retorna a instância da classe.*

---

### Método `predict(...)`

A função ``predict(...)`` realiza a previsão das classes utilizando os detectores gerados:

```python
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
```

**Parâmetros:**

* **X** (`Union[npt.NDArray, list]`): array com as características para a previsão, com **N** amostras (Linhas) e **N** colunas.

**Lança:**

* `TypeError`: Se X não for um ndarray ou uma lista.
* `FeatureDimensionMismatch`: Se o número de características em X não corresponder ao número esperado.
* `ValueError`: X contém valores que não são compostos apenas por 0 e 1.
* `ModelNotFittedError`: Se o modelo ainda não tiver sido ajustado e não possuir células de memória definidas, não conseguirá realizar predições.

**Retorna:**

* ``C``: Um array de previsão com as classes de saída para as características informadas.

---

### Método `score(...)`

A função "score(...)" calcula a precisão do modelo treinado por meio da realização de previsões e do cálculo da acurácia.

```python
def score(self, X: npt.NDArray, y: list) -> float:
```

retorna a acurácia, do tipo ``float``.

---

## Métodos privados

### Método `_assign_class_to_non_self_sample(...)`

Essa função determina a classe de uma amostra quando todos os detectores a classificam como não-própria. A classificação é realizada utilizando os métodos ``max_average_difference`` ou ``max_nearest_difference``.

```python
def _assign_class_to_non_self_sample(self, line: npt.NDArray, c: list):
```

**Parâmetros:**

* **line** (``npt.NDArray``): Amostra a ser classificada.  
* **c** (``list``): Lista de previsões para atualizar com a nova classificação.

**Retorna:**  

``npt.NDArray``: A lista de previsões `c` atualizada com a classe atribuída à amostra.
