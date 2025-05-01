# Classe BaseNSA

A classe `Base` é uma classe utilitária contendo funções com o modificador 
protected que podem ser herdadas por outras classes do módulo de seleção negativa. 
Essas funções oferecem suporte a operações comuns, como o cálculo de distâncias, 
a separação de dados para otimizar o treinamento e a previsão, além de medir a 
precisão e realizar outras tarefas necessárias.

## Funções

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