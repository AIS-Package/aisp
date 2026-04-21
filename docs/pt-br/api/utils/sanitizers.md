---
id: sanitizers
sidebar_label: sanitizers
keywords:
    - sanitize
---

# sanitizers

Funções utilitárias para validação e tratamento de parâmetros.

> **Módulo:** `aisp.utils.sanitizers`  
> **Importação:** `from aisp.utils import sanitizers`

## Funções

### sanitize_choice

```python
def sanitize_choice(value: T, valid_choices: Iterable[T], default: T) -> T:
    ...
```

Retorna o valor se estiver presente no conjunto de opções válidas; caso contrário, retorna o valor padrão.

**Parâmetros**

| Nome            | Tipo          | Padrão | Descrição                                                                     |
|-----------------|---------------|:------:|-------------------------------------------------------------------------------|
| `value`         | `T`           |   -    | O valor a ser verificado.                                                     |
| `valid_choices` | `Iterable[T]` |   -    | Uma coleção de opções válidas.                                                |
| `default`       | `T`           |   -    | O valor padrão a ser retornado se `'value'` não estiver em `'valid_choices'`. |

**Returns**

| Tipo | Descrição                                               |
|------|---------------------------------------------------------|
| `T`  | O valor original, se válido, ou o valor padrão, se não. |

---

### sanitize_param

```python
def sanitize_param(value: T, default: T, condition: Callable[[T], bool]) -> T:
    ...
```

Retorna o valor se ele satisfizer a condição especificada; caso contrário, retorna o valor padrão.

**Parâmetros**

| Nome        | Tipo                  | Padrão | Descrição                                                                               |
|-------------|-----------------------|:------:|-----------------------------------------------------------------------------------------|
| `value`     | `T`                   |   -    | O valor a ser verificado.                                                               |
| `default`   | `T`                   |   -    | O valor padrão a ser retornado se a condição não for satisfeita.                        |
| `condition` | `Callable[[T], bool]` |   -    | Uma função que recebe um valor e retorna um booleano, determinando se o valor é válido. |

**Returns**

| Tipo | Descrição                                                                    |
|------|------------------------------------------------------------------------------|
| `T`  | O valor original se a condição for satisfeita, ou o valor padrão se não for. |

---

### sanitize_seed

```python
def sanitize_seed(seed: Any) -> Optional[int]:
    ...
```

Retorna a semente se for um inteiro não negativo; caso contrário, retorna Nenhum.

**Parâmetros**

| Nome   | Tipo  | Padrão | Descrição                       |
|--------|-------|:------:|---------------------------------|
| `seed` | `Any` |   -    | O valor da seed a ser validado. |

**Returns**

| Tipo            | Descrição                                                                  |
|-----------------|----------------------------------------------------------------------------|
| `Optional[int]` | A seed original se for um inteiro não negativo, ou `None` se for inválido. |

---

### sanitize_bounds

```python
def sanitize_bounds(
    bounds: Any, problem_size: int
) -> Dict[str, npt.NDArray[np.float64]]:
    ...
```

Valida e normaliza os limites das características.

**Parâmetros**

| Nome           | Tipo  | Padrão | Descrição                                                                                                    |
|----------------|-------|:------:|--------------------------------------------------------------------------------------------------------------|
| `bounds`       | `Any` |   -    | Os limites de entrada, que devem ser None ou um dicionário com as chaves `'low'` e `'high'`.                 |
| `problem_size` | `int` |   -    | O tamanho esperado para as listas de limites normalizadas, correspondente ao número de features do problema. |

**Returns**

| Tipo              | Descrição                                                                 |
|-------------------|---------------------------------------------------------------------------|
| `Dict[str, list]` | Dicionário `{'low': [low_1, ..., low_N], 'high': [high_1, ..., high_N]}`. |

**Exceções**

| Exceção      | Descrição                                                                                                      |
|--------------|----------------------------------------------------------------------------------------------------------------|
| `TypeError`  | Se bounds não for `None` nem um dicionário com as chaves `'low'/'high'`, ou se os valores não forem numéricos. |
| `ValueError` | Se os iteráveis fornecidos tiverem tamanho incorreto.                                                          |
