# Cell Classes

Representação de células do sistema imunológico.

## Cell

Representa uma célula imune básica.

```python
@dataclass(slots=True)
class Cell:
    vector: np.ndarray
```

### Atributos
* **vector** (`np.ndarray`): Um vetor de características da célula.

### Métodos
* `__eq__(other)`: Verifica se duas células são iguais com base em seus vetores.
* `__array__()`: Interface de array para NumPy, permite que a instância seja tratada como um `np.ndarray`.
* `__getitem__(item)`: Obtém elementos do vetor de características usando indexação.

---

## BCell

Representa uma célula B de memória, derivada de `Cell`.

```python
@dataclass(slots=True, eq=False)
class BCell(Cell):
    vector: np.ndarray
```

### Métodos

### hyper_clonal_mutate(...)

```python
def hyper_clonal_mutate(
    self,
    n: int,
    feature_type: FeatureType = "continuous-features",
    bounds: Optional[npt.NDArray[np.float64]] = None
) -> npt.NDArray:
```

Clona N características das características de uma célula, gerando um conjunto de vetores mutados.

#### Parâmetros
* **n** (`int`): Número de clones a serem gerados a partir de mutações da célula original.
* **feature_type** (`FeatureType`, padrão="continuous-features"): Especifica o tipo de características com base na natureza das características de entrada.
* **bounds** (`Optional[npt.NDArray[np.float64]]`, padrão=None): Array (n_features, 2) com mínimo e máximo por dimensão.

#### Retorna
* **npt.NDArray**: Um array contendo N vetores mutados da célula original.

---

## Antibody

Representa um anticorpo com afinidade, derivado de `Cell`.

```python
@dataclass(slots=True)
class Antibody(Cell):
    vector: np.ndarray
    affinity: float
```

### Atributos
* **vector** (`npt.NDArray`): Um vetor de características da célula.
* **affinity** (`float`): Valor de afinidade do anticorpo.

### Métodos
* `__lt__(other)`: Compara este anticorpo com outro com base na afinidade.
* `__eq__(other)`: Verifica se dois anticorpos têm a mesma afinidade.

---

## Detector

Representa um detector não-próprio da classe RNSA.

```python
@dataclass(slots=True)
class Detector:
    position: npt.NDArray[np.float64]
    radius: Optional[float] = None
```

### Atributos
* **position** (`npt.NDArray[np.float64]`): Vetor de características do detector.
* **radius** (`Optional[float]`): Raio do detector, usado no algoritmo V-detector.

---