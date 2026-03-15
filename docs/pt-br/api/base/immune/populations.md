---
id: populations
sidebar_label: populations
keywords:
  - binário
  - classificação
  - limiar de afinidade
  - real-valor
  - célula-b de memória
  - expansão clonal
  - população
---

# populations

Fornece funções utilitárias para **gerar populações** de anticorpos utilizadas em algoritmos imuno-inspirados.

> **Módulo:** `aisp.base.immune`  
> **Importação:** `from aisp.base.immune import populations`

## Funções

### generate_random_antibodies

```python
def generate_random_antibodies(
    n_samples: int,
    n_features: int,
    feature_type: FeatureTypeAll = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None,
) -> npt.NDArray:
    ...
```

Gera uma população aleatória de anticorpos.

**Parâmetros**

| Nome           | Tipo                                                    |         Padrão          | Descrição                                                                                                                      |
|----------------|---------------------------------------------------------|:-----------------------:|--------------------------------------------------------------------------------------------------------------------------------|
| `n_samples`    | `int`                                                   |            -            | Número de anticorpos (amostras) que serão gerados.                                                                             |
| `n_features`   | `int`                                                   |            -            | Número de características (dimensões) para cada anticorpo.                                                                     |
| `feature_type` | [`FeatureTypeAll`](../../utils/types.md#featuretypeall) | `"continuous-features"` | Especifica o tipo das características: "continuous-features", "binary-features", "ranged-features", or "permutation-features". |
| `bounds`       | `npt.NDArray[np.float64]`                               |         `None`          | Array (n_features, 2) contendo os valores mínimo e máximo por dimensão.                                                        |

**Exceções**

`ValueError` - Lançado caso o número de características seja menor ou igual a zero.

**Returns**

`npt.NDArray` - Array de dimensão (n_samples, n_features) contendo os anticorpos gerados.
