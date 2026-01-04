# Seleção Negativa

As funções realizam verificações de detectores e utilizam decoradores Numba para compilação Just-In-Time.

## Função `check_detector_bnsa_validity(...)`

```python
def check_detector_bnsa_validity(
    x_class: npt.NDArray,
    vector_x: npt.NDArray,
    aff_thresh: float
) -> bool:
```

Verifica a validade de um candidato a detector (vector_x) contra amostras de uma classe (x_class) usando a distância de Hamming. Um detector é considerado INVÁLIDO se a sua distância para qualquer amostra em ``x_class`` for menor ou igual a ``aff_thresh``.

**Os parâmetros de entrada são:**

* **x_class** (``npt.NDArray``): Array contendo as amostras da classe. Formato esperado: (n_amostras, n_características).
* **vector_x** (``npt.NDArray``): Array representando o detector. Formato esperado: (n_características,).
* **aff_thresh** (``float``): Limiar de afinidade.

**Retorna:**

* True se o detector for válido, False caso contrário.

---

## Função `bnsa_class_prediction(...)`

```python
def bnsa_class_prediction(
    features: npt.NDArray,
    class_detectors: npt.NDArray,
    aff_thresh: float
) -> int:
```

Define a classe de uma amostra a partir dos detectores não-próprios.

**Os parâmetros de entrada são:**

* **features** (``npt.NDArray``): amostra binária a ser classificada (shape: [n_features]).
* **class_detectors** (``npt.NDArray``): Matriz contendo os detectores de todas as classes (shape: [n_classes, n_detectors, n_features]).
* **aff_thresh** (``float``): Limiar de afinidade que determina se um detector reconhece a amostra como não-própria.

**Retorna:**

* `int`: Índice da classe predita. Retorna -1 se for não-própria para todas as classes.

---

## Função `check_detector_rnsa_validity(...)`

```python
def check_detector_rnsa_validity(
    x_class: npt.NDArray,
    vector_x: npt.NDArray,
    threshold: float,
    metric: int,
    p: float
) -> bool:
```

Verifica a validade de um candidato a detector (vector_x) contra amostras de uma classe (x_class) usando a distância de Hamming. Um detector é considerado INVÁLIDO se a sua distância para qualquer amostra em ``x_class`` for menor ou igual a ``aff_thresh``.

**Os parâmetros de entrada são:**

* **x_class** (``npt.NDArray``): Array contendo as amostras da classe. Formato esperado: (n_amostras, n_características).
* **vector_x** (``npt.NDArray``): Array representando o detector. Formato esperado: (n_características,).
* **threshold** (``float``): Afinidade.
* **metric** (``int``): Métrica de distância a ser utilizada. Opções disponíveis: 0 (Euclidean), 1 (Manhattan), 2 (Minkowski).
* **p** (``float``): Parâmetro da métrica de Minkowski (utilizado apenas se `metric` for "minkowski").

**Retorna:**

* `True` se o detector for válido, `False` caso contrário.

---
