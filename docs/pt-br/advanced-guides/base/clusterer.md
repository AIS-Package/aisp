# BaseClusterer

Classe base abstrata para algoritmos de clustering.

Esta classe define a interface central para modelos de agrupamento. Ela exige
a implementação dos métodos **`fit`** e **`predict`** em todas as classes derivadas,
e fornece uma implementação padrão para **`fit_predict`** e **`get_params`**.

---

### Função fit(...)

```python
def fit(self, X: npt.NDArray, verbose: bool = True) -> BaseClusterer
```

Ajusta o modelo aos dados de treinamento.
Este método abstrato deve ser implementado pelas subclasses.

**Parâmetros**:

* ***X***: `npt.NDArray` - Dados de entrada utilizados para treinar o modelo.
* ***verbose***: `bool`, default=True - Indica se a saída detalhada durante o treinamento deve ser exibida.

**Retorna**:

* ***self***: `BaseClusterer` - Instância da classe que implementa este método.

**Implementação**:

* [AiNet](../../../classes/Immune%20Network%20Theory/AiNet.md#function-fit)

---

### Função predict(...)

```python
def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]
```

Gera previsões com base nos dados de entrada.
Este método abstrato deve ser implementado pelas subclasses.

**Parâmetros**:

* ***X***: `npt.NDArray` - Dados de entrada para os quais as previsões serão geradas.

**Retorna**:

* ***predictions***: `Optional[npt.NDArray]` - Rótulos previstos dos clusters para cada amostra de entrada, ou `None` caso a previsão não seja possível.

**Implementação**:

* [AiNet](../../../classes/Immune%20Network%20Theory/AiNet.md#function-predict)

---

### Função fit_predict(...)

```python
def fit_predict(self, X: npt.NDArray, verbose: bool = True) -> Optional[npt.NDArray]
```

Método de conveniência que combina `fit` e `predict` em uma única chamada.

**Parâmetros**:

* ***X***: `npt.NDArray` - Dados de entrada para os quais as previsões serão geradas.
* ***verbose***: `bool`, default=True - Indica se a saída detalhada durante o treinamento deve ser exibida.

**Retorna**:

* ***predictions***: `Optional[npt.NDArray]` - Rótulos previstos dos clusters para cada amostra de entrada, ou `None` caso a previsão não seja possível.