# Classe BaseNSA

A classe `Base` é uma classe utilitária contendo funções com o modificador 
protected que podem ser herdadas por outras classes do módulo de seleção negativa. 
Essas funções oferecem suporte a operações comuns, como o cálculo de distâncias, 
a separação de dados para otimizar o treinamento e a previsão, além de medir a 
precisão e realizar outras tarefas necessárias.

## Funções

## Funções Protegidas:

---

### def _check_and_raise_exceptions_fit(...)

```python
def _check_and_raise_exceptions_fit(
    X: npt.NDArray,
    y: npt.NDArray,
    _class_: Literal["RNSA", "BNSA"] = "RNSA",
) -> None:
```
Função responsável por verificar os parâmetros da função fit e lançar exceções se a verificação não for bem-sucedida.

**Parâmetros**:
* **X** (``npt.NDArray``): Array de treinamento, contendo as amostras e suas características, [``N samples`` (linhas)][``N features`` (colunas)].
* ***y*** (``npt.NDArray``): Array de classes alvo de ``X`` com [``N samples`` (linhas)].
* ***_class_*** (Literal[RNSA, BNSA], opcional): Classe atual. O padrão é 'RNSA'.


**Lança:**
* ``TypeError``: Se X ou y não forem ndarrays ou tiverem formas incompatíveis.
* ``MaxDiscardsReachedError``: O número máximo de descartes do detector foi atingido durante
a maturação. Verifique o valor do raio definido e considere reduzi-lo.


### def _check_and_raise_exceptions_predict(...)

```python
def _check_and_raise_exceptions_predict(
    X: npt.NDArray,
    expected: int = 0,
    _class_: Literal["RNSA", "BNSA"] = "RNSA",
) -> None:
```
Função responsável por verificar os parâmetros da função predict e lançar exceções caso a verificação não seja bem-sucedida.
**Parâmetros**:
* ***X*** (``npt.NDArray``): Array de treinamento, contendo as amostras e suas características, [``N samples`` (linhas)][``N features`` (colunas)].
* ***expected*** (``int``): Número esperado de características por amostra (colunas em X).
* ***_class_*** (``Literal[RNSA, BNSA], opcional``): Classe atual. O padrão é 'RNSA'.

**Lança:**
* ``TypeError``: Se X ou y não forem ndarrays ou tiverem formas incompatíveis.
* ``MaxDiscardsReachedError``: O número máximo de descartes do detector foi atingido durante
a maturação. Verifique o valor do raio definido e considere reduzi-lo.
