# Classe nsa._Base

A classe `Base` é uma classe utilitária contendo funções com o modificador 
protected que podem ser herdadas por outras classes do módulo de seleção negativa. 
Essas funções oferecem suporte a operações comuns, como o cálculo de distâncias, 
a separação de dados para otimizar o treinamento e a previsão, além de medir a 
precisão e realizar outras tarefas necessárias.

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

## Métodos abstratos

### def fit(...)

```python
def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True)
```

Ajusta o modelo aos dados de treinamento.

Implementação:

- [RNSA](../../classes/Negative%20Selection/RNSA.md#função-fit)
- [BNSA](../../classes/Negative%20Selection/BNSA.md#função-fit)

### def predict(...)

```python
def predict(self, X) -> Optional[npt.NDArray]:
```

Realiza a previsão dos rótulos para os dados fornecidos.

Implementação:

- [RNSA](../../classes/Negative%20Selection/RNSA.md#função-predict)
- [BNSA](../../classes/Negative%20Selection/BNSA.md#função-predict)