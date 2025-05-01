Classe base para algoritmo de classificação.

# BaseClassifier

Classe base para algoritmos de classificação, definindo os métodos abstratos ``fit`` e ``predict``, e implementando o método ``get_params``.

## Funções

### def score(...)

```python
def score(self, X: npt.NDArray, y: list) -> float
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

### def get_params(...)

```python
def get_params(self, deep: bool = True) -> dict:
```
A função get_params retorna um dicionário com os parâmetros principais do objeto.

Esta função é necessária para garantir a compatibilidade com as funções do scikit-learn.

---

## Métodos abstratos

### def fit(...)

```python
def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True)
```

Ajusta o modelo aos dados de treinamento.

Implementação:

- [RNSA](../../../classes/Negative%20Selection/RNSA.md#função-fit)
- [BNSA](../../../classes/Negative%20Selection/BNSA.md#função-fit)

### def predict(...)

```python
def predict(self, X) -> Optional[npt.NDArray]:
```

Realiza a previsão dos rótulos para os dados fornecidos.

Implementação:

- [RNSA](../../../classes/Negative%20Selection/RNSA.md#função-predict)
- [BNSA](../../../classes/Negative%20Selection/BNSA.md#função-predict)

# Classe Detector

Representa um detector não-próprio do class RNSA.

Atributos
----------
* ***position*** (``np.ndarray``): Vetor de características do detector.
* ***radius*** (``float, opcional``): Raio do detector, utilizado no algoritmo V-detector.