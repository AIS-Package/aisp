Classe base para algoritmo de classificação.

# BaseClassifier

Classe base para algoritmos de classificação, definindo os métodos abstratos ``fit`` e ``predict``.

## Métodos

### Método `score(...)`

```python
def score(self, X: npt.NDArray, y: list) -> float
```

A função de pontuação (score) calcula a precisão da previsão.

Esta função realiza a previsão de X e verifica quantos elementos são iguais entre o vetor y e y_predicted.
Esta função foi adicionada para compatibilidade com algumas funções do scikit-learn.

**Parâmetros**:

* **X** (`npt.NDArray`): Conjunto de características com formato (n_amostras, n_características).
* **y** (`list`): Valores verdadeiros com formato (n_amostras,).

**Retorna:**

* precisão (`float`): A precisão do modelo.

---

### Método `_slice_index_list_by_class(...)`

A função ``_slice_index_list_by_class(...)``, separa os índices das linhas conforme a classe de saída, para percorrer o array de amostra, apenas nas posições que a saída for a classe que está sendo treinada:

```python
def _slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Retorna um dicionário com as classes como chave e os índices em ``X`` das amostras.

---

## Métodos abstratos

### Método `fit(...)`

```python
@abstractmethod
def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True) -> BaseClassifier:
```

Ajusta o modelo aos dados de treinamento.

Implementação:

* [RNSA](../../classes/Negative%20Selection/RNSA.md#método-fit)
* [BNSA](../../classes/Negative%20Selection/BNSA.md#método-fit)
* [AIRS](../../classes/Clonal%20Selection%20Algorithms/AIRS.md#método-fit)

### Método `predict(...)`

```python
@abstractmethod
def predict(self, X) -> Optional[npt.NDArray]:
```

Realiza a previsão dos rótulos para os dados fornecidos.

Implementação:

* [RNSA](../../classes/Negative%20Selection/RNSA.md#método-predict)
* [BNSA](../../classes/Negative%20Selection/BNSA.md#método-predict)
* [AIRS](../../classes/Clonal%20Selection%20Algorithms/AIRS.md#método-predict)
