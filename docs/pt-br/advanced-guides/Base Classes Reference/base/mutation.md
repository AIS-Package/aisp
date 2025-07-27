# Mutation

Contém funções que geram conjuntos de clones mutados a partir de vetores contínuos ou binários, simulando o processo de expansão clonal em sistemas imunológicos artificiais.

## clone_and_mutate_continuous

```python
@njit([(types.float64[:], types.int64)], cache=True)
def clone_and_mutate_continuous(
    vector: npt.NDArray[np.float64],
    n: int
) -> npt.NDArray[np.float64]:
```

Gera um conjunto de clones mutados a partir de um vetor contínuo.

Esta função cria `n` clones do vetor de entrada e aplica mutações aleatórias em cada um, simulando o processo de expansão clonal em sistemas imunes artificiais. Cada clone recebe um número aleatório de mutações em posições distintas do vetor original.

### Parâmetros

* `vector` (`npt.NDArray[np.float64]`): Vetor contínuo original que representa a célula imune a ser clonada e mutada.
* `n` (`int`): Quantidade de clones mutados a serem gerados.

### Retorno

* `clone_set` (`npt.NDArray[np.float64]`): Array com forma `(n, len(vector))` contendo os `n` clones mutados do vetor original.

---

## clone_and_mutate_binary

```python
@njit([(types.boolean[:], types.int64)], cache=True)
def clone_and_mutate_binary(
    vector: npt.NDArray[np.bool_],
    n: int
) -> npt.NDArray[np.bool_]:
```

Gera um conjunto de clones mutados a partir de um vetor binário.

Esta função cria `n` clones do vetor binário de entrada e aplica mutações aleatórias em alguns bits, simulando a expansão clonal em sistemas imunes artificiais com representações discretas.

### Parâmetros

* `vector` (`npt.NDArray[np.bool_]`): Vetor binário original que representa a célula imune a ser clonada e mutada.
* `n` (`int`): Quantidade de clones mutados a serem gerados.

### Retorno

* `clone_set` (`npt.NDArray[np.bool_]`): Array com forma `(n, len(vector))` contendo os `n` clones mutados do vetor original.

---

## clone_and_mutate_ranged

```python
@njit([(types.float64[:], types.int64, types.float64[:, :])], cache=True)
def clone_and_mutate_ranged(
vector: npt.NDArray[np.float64],
n: int,
bounds: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
```

Gera um conjunto de clones mutados a partir de um vetor contínuo usando limites personalizados por dimensão.

Esta função cria `n` clones do vetor de entrada e aplica mutações aleatórias em cada um, simulando o processo de expansão clonal em sistemas imunes artificiais. Cada clone recebe um número aleatório de mutações em posições distintas do vetor original.

### Parâmetros

* `vector` (`npt.NDArray[np.float64]`): Vetor contínuo original que representa a célula imune a ser clonada e mutada.
* `n` (`int`): Quantidade de clones mutados a serem gerados.
* `bounds` (`npt.NDArray[np.float64]`): Um array 2D com o formato `(len(vector), 2)` contendo os valores mínimo e máximo para cada dimensão.

### Retorna

* `clone_set` (`npt.NDArray[np.float64]`): Array com forma `(n, len(vector))` contendo os `n` clones mutados do vetor original.
