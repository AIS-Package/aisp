# BaseClusterer

Classe base abstrata para algoritmos de clustering.

Esta classe define a interface central para modelos de agrupamento. Ela exige
a implementação dos métodos **`fit`** e **`predict`** em todas as classes derivadas,
e fornece uma implementação padrão para **`fit_predict`** e **`get_params`**.

---

## Métodos abstratos

### Método `fit(...)`

```python
@abstractmethod
def fit(self, X: Union[npt.NDArray, list], verbose: bool = True) -> BaseClusterer:
```

Ajusta o modelo aos dados de treinamento.
Este método abstrato deve ser implementado pelas subclasses.

**Parâmetros:**

* **X** (`Union[npt.NDArray, list]`): Dados de entrada utilizados para treinar o modelo.
* **verbose** (`bool`): default=True - Indica se a saída detalhada durante o treinamento deve ser exibida.

**Retorna:**

* **self:** `BaseClusterer` - Instância da classe que implementa este método.

**Implementações:**

* [AiNet](../../../classes/Immune%20Network%20Theory/AiNet.md#método-fit)

---

### Método `predict(...)`

```python
@abstractmethod
def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
```

Gera previsões com base nos dados de entrada.
Este método abstrato deve ser implementado pelas subclasses.

**Parâmetros:**

* **X** (`Union[npt.NDArray, list]`): Dados de entrada para os quais as previsões serão geradas.

**Retorna:**

* **predictions** (`npt.NDArray`): Rótulos previstos dos clusters para cada amostra de entrada.

**Implementações:**

* [AiNet](../../../classes/Immune%20Network%20Theory/AiNet.md#método-predict)

---

## Métodos

### Método `fit_predict(...)`

```python
def fit_predict(self, X: Union[npt.NDArray, list], verbose: bool = True) -> npt.NDArray:
```

Método de conveniência que combina `fit` e `predict` em uma única chamada.

**Parâmetros:**

* **X** (`Union[npt.NDArray, list]`): Dados de entrada para os quais as previsões serão geradas.
* **verbose** (`bool`, default=True): Indica se a saída detalhada durante o treinamento deve ser exibida.

**Retorna:**

* **predictions** (`npt.NDArray`): Rótulos previstos dos clusters para cada amostra de entrada.