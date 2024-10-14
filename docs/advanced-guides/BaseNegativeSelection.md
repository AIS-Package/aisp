# Português


# Classe NSA._base

A classe `Base` é uma classe utilitária contendo funções com o modificador 
protected que podem ser herdadas por outras classes do módulo de seleção negativa. 
Essas funções oferecem suporte a operações comuns, como o cálculo de distâncias, 
a separação de dados para otimizar o treinamento e a previsão, além de medir a 
precisão e realizar outras tarefas necessárias.

## Funções

### def fit(...)

```python
@abstractmethod
def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True)
```

Função para treinar o modelo usando os dados de entrada ``X`` e os classes correspondentes ``y``.

Este método abstrato é implementado pela classe que o herdar.

**Parâmetros:**
* **_X_** (``npt.NDArray``): Dados de entrada utilizados para o treinamento do modelo, previamente \
    normalizados no intervalo [0, 1].
* **_y_** (``npt.NDArray``): Rótulos ou valores-alvo correspondentes aos dados de entrada.
* **_verbose_** (``bool``, opcional): Flag para ativar ou desativar a saída detalhada durante o \
    treinamento. O padrão é ``True``.

Retornos:
---
* `self`: Retorna a instância da classe que implementa este método.

---

### def predict(...)

```python
@abstractmethod
def predict(self, X) -> Optional[npt.NDArray]
```
Função para gerar previsões com base nos dados de entrada ``X``.

Este método abstrato é implementado pela classe que o herdar.

**Parâmetros:**

* ***X*** (``npt.NDArray``): Dados de entrada para os quais as previsões serão geradas.

**Retorna:**


* ***Previsões*** (``Optional[npt.NDArray]``): Valores previstos para cada amostra de entrada, ou ``None`` se a previsão falhar.


## Funções Protegidas:

---

### def _distance(...):

```python
def _distance(self, u: npt.NDArray, v: npt.NDArray)
```

Função para calcular a distância entre dois pontos usando a "métrica" escolhida.

**Parâmetros**:
* ***u*** (``npt.NDArray``): Coordenadas do primeiro ponto.
* ***v*** (``npt.NDArray``): Coordenadas do segundo ponto.

**Retorna**:
* Distância (``double``) entre os dois pontos.

---

### def _check_and_raise_exceptions_fit(...)

```python
def _check_and_raise_exceptions_fit(self, X: npt.NDArray = None, y: npt.NDArray = None, _class_: Literal['RNSA', 'BNSA'] = 'RNSA')
```
Função responsável por verificar os parâmetros da função fit e lançar exceções se a verificação não for bem-sucedida.

**Parâmetros**:
* **X** (``npt.NDArray``): Array de treinamento, contendo as amostras e suas características, [``N samples`` (linhas)][``N features`` (colunas)].
* ***y*** (``npt.NDArray``): Array de classes alvo de ``X`` com [``N samples`` (linhas)].
* ***_class_*** (Literal[RNSA, BNSA], opcional): Classe atual. O padrão é 'RNSA'.

---


### def _score(...)

```python
def _score(self, X: npt.NDArray, y: list) -> float
```

A função de pontuação (score) calcula a precisão da previsão.

Esta função realiza a previsão de X e verifica quantos elementos são iguais entre o vetor y e y_predicted. 
Esta função foi adicionada para compatibilidade com algumas funções do scikit-learn.

**Parâmetros**:
+ ***X***: np.ndarray
    Conjunto de características com formato (n_amostras, n_características).
+ ***y***: np.ndarray
    Valores verdadeiros com formato (n_amostras,).

**Retorna**:

+ precisão: float
    A precisão do modelo.

---

# English

# NSA._base Class

The ``_Base`` class contains utility functions with the ``protected`` modifier that can be inherited by various classes for ease of use. It includes functions for distance calculation, data separation to improve training and prediction efficiency, accuracy measurement and other functions.

## Protected Functions:

---

### def _distance(...):

```python
def _distance(self, u: npt.NDArray, v: npt.NDArray)
```

Function to calculate the distance between two points by the chosen ``metric``.

**Parameters**:
* ***u*** (``npt.NDArray``): Coordinates of the first point.
* ***v*** (``npt.NDArray``): Coordinates of the second point.

**returns**:
* Distance (``double``) between the two points.

---

### def _check_and_raise_exceptions_fit(...):
```python
def _check_and_raise_exceptions_fit(self, X: npt.NDArray = None, y: npt.NDArray = None, _class_: Literal['RNSA', 'BNSA'] = 'RNSA')
```
Function responsible for verifying fit function parameters and throwing exceptions if the verification is not successful.

**Parameters**:
* ***X*** (``npt.NDArray``): Training array, containing the samples and their characteristics, [``N samples`` (rows)][``N features`` (columns)].
* ***y*** (``npt.NDArray``): Array of target classes of ``X`` with [``N samples`` (lines)].
* ***_class_*** (Literal[RNSA, BNSA], optional): Current class. Defaults to 'RNSA'.

---

### def _slice_index_list_by_class(...)

```python
def _slice_index_list_by_class(self, y: npt.NDArray) -> dict
```

The function ``__slice_index_list_by_class(...)``, separates the indices of the lines according to the output class, to loop through the sample array, only in positions where the output is the class being trained.

**Parameters**:
* ***y*** (`npt.NDArray`): Receives a ``y``[``N sample``] array with the output classes of the ``X`` sample array.

**returns**:
* dict: A dictionary with the list of array positions(``y``), with the classes as key.

---

### def _score(...)

```python
def _score(self, X: npt.NDArray, y: list) -> float
```
Score function calculates forecast accuracy.

This function performs the prediction of X and checks how many elements are equal between vector y and y_predicted. 
This function was added for compatibility with some scikit-learn functions.

**Parameters**:
+ ***X*** `np.ndarray`:
    Feature set with shape (n_samples, n_features).
+ ***y*** ` np.ndarray`:
    True values with shape (n_samples,).

**Returns**:

+ accuracy `float`:
    The accuracy of the model.

### def fit(...)

```python
@abstractmethod
def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True)
```

Function to train the model using the input data ``X`` and corresponding labels ``y``.

This abstract method is implemented by the class that inherits it.

**Parameters:**

* **_X_** (``npt.NDArray``): Input data used for training the model, previously normalized to the range [0, 1].
* **_y_** (``npt.NDArray``): Corresponding labels or target values for the input data.
* **_verbose_** (``bool``, optional): Flag to enable or disable detailed output during \
        training. Default is ``True``.

Returns:
---
* `self`: Returns the instance of the class that implements this method.

---

### def predict(...)

```python
@abstractmethod
def predict(self, X) -> Optional[npt.NDArray]
```

Function to generate predictions based on the input data ``X``.

This abstract method is implemented by the class that inherits it.

**Parameters:**

* ***X*** (``npt.NDArray``): Input data for which predictions will be generated.

**Returns:**

* ***Predictions*** (``Optional[npt.NDArray]``): Predicted values for each input sample, or ``None`` if the prediction fails.
