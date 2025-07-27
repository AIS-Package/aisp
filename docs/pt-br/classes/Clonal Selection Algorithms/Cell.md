## Classe Cell:

Represents a memory B-cell.

### Construtor:

Parâmetros:

* vector (`Optional[npt.NDArray]`): Vetor de características da célula. Padrão é None.


---

### Função hyper_clonal_mutate(...):

Parâmetros:

* n (`int`): O número de clones a serem gerados a partir de mutações da célula original.
* feature_type (`Literal["continuous-features", "binary-features", "ranged-features"]`): Especifica o tipo de algoritmo com base na natureza das características de entrada.
* bounds (``np.ndarray``): Array (n_features, 2) com mínimo e máximo por dimensão.
```python
def hyper_clonal_mutate(
    self,
    n: int,
    feature_type: Literal[
        "binary-features",
        "continuous-features",
        "ranged-features"
    ],
    bounds: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray
```

Retorna um array contendo N vetores mutados a partir da célula original.
