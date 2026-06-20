---
id: progress-bar
sidebar_label: ProgressBar
---


# ProgressBar

Exibe uma barra de progresso no console para acompanhar a execução de um algoritmo.

Esta classe monta uma barra de progresso no formato `{description}{bar*slots}{suffix}` e, ao chamar a função
`finish`, exibe a barra final com o tempo total desde a instanciação da classe.


> **Módulo:** `aisp.utils.display`  
> **Importação:** `from aisp.utils.display import ProgressBar`

---

## Constructor Parameters

| Name          | Type   | Default | Description                                            |
|---------------|--------|:-------:|--------------------------------------------------------|
| `total`       | `int`  |    -    | Número total de interações.                            |
| `suffix`      | `str`  |    -    | Texto opcional exibido apos a barra de progresso.      |
| `description` | `str`  |    -    | Texto opcional exibido antes a barra de progresso.     |
| `slots`       | `int`  |    -    | Numero de caracteres para formar a barra de progresso. |
| `verbose`     | `bool` |    -    | Se False, não imprime nada no terminal.                |

**Raises**

| Exception    | Description                                          |
|--------------|------------------------------------------------------|
| `ValueError` | Se `total` ou `slots` forem menores ou igual a zero. |

---

## Public Methods

### set_description

```python
def set_description(self, description: str) -> None:
    ...
```

Atualize o texto da descrição antes da barra de progresso.

**Parameters**

| Name          | Type  | Default | Description    |
|---------------|-------|:-------:|----------------|
| `description` | `str` |    -    | Nova descrição |

### update

```python
def update(self, increment: int = 1) -> None:
    ...
```

Atualiza a barra de progresso, incrementando seu valor pela quantidade informada.

**Parameters**

| Name        | Type  | Default | Description                                                       |
|-------------|-------|:-------:|-------------------------------------------------------------------|
| `increment` | `int` |   `1`   | Número de iterações concluídas para adicionar ao progresso atual. |


**Raises**

| Exception    | Description                    |
|--------------|--------------------------------|
| `ValueError` | Se o `increment` for negativo. |

### finish

```python
def finish(self) -> None:
    ...
```

Encerre a exibição da barra de progresso e imprima o tempo total.


