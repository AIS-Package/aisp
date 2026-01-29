# Populations Module

Fornece funções utilitárias para gerar populações de anticorpos em algoritmos imunológicos.

# generate_random_antibodies(...)

```python
def generate_random_antibodies(
    n_samples: int,
    n_features: int,
    feature_type: FeatureTypeAll = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray:
```

Gera uma população aleatória de anticorpos

**Parâmetros**:
* **n_samples** (`int`): Número de anticorpos (amostras) a serem gerados.
* **n_features** (`int`): Número de características (dimensões) de cada anticorpo.
* **feature_type** (`FeatureTypeAll`, default="continuous-features"): Especifica o tipo de características:: "continuous-features", "binary-features", "ranged-features", or "permutation-features".
* **bounds** (`Optional[npt.NDArray[np.float64]]`, default=None): Array (2, n_features) com valores mínimo e máximo por dimensão.

**Returns**:
* **npt.NDArray**: Array com forma (n_samples, n_features) contendo os anticorpos gerados.

**Raises**:
* **ValueError**: Se o número de características for menor que 0. 