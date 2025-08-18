# BaseAiNet(BaseClusterer, ABC)

Classe base para algoritmos de Teoria de Redes baseados em AiNet.

A classe base contém funções utilizadas por múltiplas classes no pacote AiNet e
são consideradas essenciais para o funcionamento adequado de algoritmos de clustering baseados na teoria de redes imunes.

---

### def _check_and_raise_exceptions_fit(...)

Verifica os parâmetros do método `fit` e lança exceções caso a verificação não seja bem-sucedida.

```python
@staticmethod
def _check_and_raise_exceptions_fit(X: npt.NDArray)
```

**Parâmetros**:

* ***X*** (`npt.NDArray`): Matriz de treinamento contendo as amostras e suas características, \[`N amostras` (linhas)]\[`N atributos` (colunas)].

**Exceções**:

* `TypeError`: Se X não for um `ndarray` ou uma `list`.

---

### def _check_and_raise_exceptions_predict(...)

Verifica os parâmetros do método `predict` e lança exceções caso a verificação não seja bem-sucedida.

```python
@staticmethod
def _check_and_raise_exceptions_predict(
    X: npt.NDArray,
    expected: int = 0,
    feature_type: FeatureType = "continuous-features"
) -> None
```

**Parâmetros**:

* ***X*** (`npt.NDArray`): Matriz de entrada para predição, contendo as amostras e suas características, \[`N amostras` (linhas)]\[`N atributos` (colunas)].
* ***expected*** (`int`, default=0): Número esperado de atributos por amostra (colunas em X).
* ***feature_type*** (`FeatureType`, default="continuous-features"): Especifica o tipo de atributos: `"continuous-features"`, `"binary-features"` ou `"ranged-features"`.

**Exceções**:

* `TypeError`: Se X não for um `ndarray` ou uma `list`.
* `FeatureDimensionMismatch`: Se o número de atributos em X não corresponder ao esperado.
* `ValueError`: Se `feature_type` for `"binary-features"` e X contiver valores diferentes de 0 e 1.

---

### def _generate_random_antibodies(...)

Gera uma população aleatória de anticorpos.

```python
@staticmethod
def _generate_random_antibodies(
    n_samples: int,
    n_features: int,
    feature_type: FeatureType = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray
```

**Parâmetros**:

* ***n_samples*** (`int`): Número de anticorpos (amostras) a serem gerados.
* ***n_features*** (`int`): Número de atributos (dimensões) para cada anticorpo.
* ***feature_type*** (`FeatureType`, default="continuous-features"): Especifica o tipo de atributos: `"continuous-features"`, `"binary-features"` ou `"ranged-features"`.
* ***bounds*** (`Optional[npt.NDArray[np.float64]]`): Matriz de formato `(n_features, 2)` com valores mínimos e máximos por dimensão (usado apenas para atributos do tipo "ranged").

**Retorna**:

* `npt.NDArray`: Matriz de formato `(n_samples, n_features)` contendo os anticorpos gerados.
  O tipo dos dados depende de `feature_type` (float para contínuo/ranged, bool para binário).

**Exceções**:

* `ValueError`: Se `n_features <= 0`.

---

### Métodos Abstratos

#### fit(...)

```python
def fit(self, X: npt.NDArray, verbose: bool = True) -> "BaseAiNet"
```

Treina o modelo de clustering AiNet.
Deve ser implementado pelas subclasses.

**Parâmetros**:

* ***X*** (`npt.NDArray`): Dados de entrada utilizados para treinar o modelo.
* ***verbose*** (`bool`, default=True): Indica se a saída detalhada durante o treinamento deve ser exibida.

**Retorna**:

* `BaseAiNet`: Instância da classe que implementa o método.

---

#### predict(...)

```python
def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]
```

Gera predições de clusters para os dados de entrada.
Deve ser implementado pelas subclasses.

**Parâmetros**:

* ***X*** (`npt.NDArray`): Dados de entrada para os quais as predições serão geradas.

**Retorna**:

* `Optional[npt.NDArray]`: Rótulos previstos de cluster para cada amostra de entrada, ou `None` caso a predição não seja possível.
