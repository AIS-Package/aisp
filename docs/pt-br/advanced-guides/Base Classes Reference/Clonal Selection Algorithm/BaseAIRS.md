# BaseAIRS(BaseClassifier, ABC)

Classe base para o algoritmo **AIRS**.

A classe base contém funções que são utilizadas por mais de uma classe no pacote e, portanto, são consideradas essenciais para o funcionamento geral do sistema.

---

### def _check_and_raise_exceptions_fit(...):

Verifica os parâmetros de ajuste (*fit*) e lança exceções caso a verificação não seja bem-sucedida.

```python
@staticmethod
def _check_and_raise_exceptions_fit(
    X: npt.NDArray = None,
    y: npt.NDArray = None,
    algorithm: Literal[
        "continuous-features", "binary-features"
    ] = "continuous-features"
):
```

**Parâmetros**:

* ***X*** (`npt.NDArray`): Array de treinamento, contendo as amostras e suas características, com formato [`N amostras` (linhas)][`N características` (colunas)].
* ***y*** (`npt.NDArray`): Array das classes alvo de `X` com [`N amostras` (linhas)].
* ***algorithm*** (`Literal["continuous-features", "binary-features"], opcional`): Especifica o tipo de algoritmo a ser usado, dependendo se os dados de entrada possuem características contínuas ou binárias.

**Exceções**:

* `TypeError`:
  Se X ou y não forem ndarrays ou tiverem formatos incompatíveis.
* `ValueError`:
  Se a *classe* for BNSA e X contiver valores que não sejam compostos apenas por 0 e 1.

---

### def _check_and_raise_exceptions_predict(...):

Verifica os parâmetros de predição e lança exceções caso a verificação não seja bem-sucedida.

```python
@staticmethod
def _check_and_raise_exceptions_predict(
    X: npt.NDArray = None,
    expected: int = 0,
    algorithm: Literal[
        "continuous-features", "binary-features"
    ] = "continuous-features"
) -> None:
```

**Parâmetros**:

* ***X*** (`npt.NDArray`): Array de entrada, contendo as amostras e suas características, com formato [`N amostras` (linhas)][`N características` (colunas)].
* ***expected*** (`int`): Número esperado de características por amostra (colunas de X).
* ***algorithm*** (`Literal["continuous-features", "binary-features"], opcional`): Especifica o tipo de algoritmo a ser usado, dependendo se os dados de entrada possuem características contínuas ou binárias.

**Exceções**:

* `TypeError`:
  Se X não for um ndarray ou lista.
* `FeatureDimensionMismatch`:
  Se o número de características em X não corresponder ao número esperado.
* `ValueError`:
  Se o algoritmo for "binary-features" e X contiver valores que não sejam compostos apenas por 0 e 1.

---