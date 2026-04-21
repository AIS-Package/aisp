---
id: types
sidebar_label: types
keywords:
    - tipos
    - tipagem
    - aliases
    - type hints
---

# types

Define aliases de tipos usados em todo o pacote para melhorar a legibilidade.

> **Módulo:** `aisp.utils.types`  
> **Importação:** `from aisp.utils import types`

## Type aliases

### FeatureType

```python
FeatureType: TypeAlias = Literal[
    "binary-features", "continuous-features", "ranged-features"
]
```

Tipo das características de entrada

- `"binary-features"`: Valores apenas com (0 ou 1, `False` ou `True`).
- `"continuous-features"`: Valores numéricos contínuos.
- `"ranged-features"`: Valores entre um intervalo definido.

---

### FeatureTypeAll

```python
FeatureTypeAll: TypeAlias = Union[FeatureType, Literal["permutation-features"]]
```

Mesmo que `FeatureType`, mais:

- `"permutation-features"`: valores representados como permutação.

---

### MetricType

```python
MetricType: TypeAlias = Literal["manhattan", "minkowski", "euclidean"]
```

Métrica de distância utilizada nos cálculos:

- `"manhattan"`: distância de Manhattan entre dois pontos.
- `"minkowski"`: distância de Minkowski entre dois pontos.
- `"euclidean"`: distância de Euclidean entre dois pontos.

---