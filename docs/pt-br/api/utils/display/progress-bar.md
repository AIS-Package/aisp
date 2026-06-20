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

### method_name

```python
def method_name(
    param_1: type,
) -> type:
    ...
```

Description.

**Parameters**

| Name      | Type   | Default | Description               |
|-----------|--------|:-------:|---------------------------|
| `param_1` | `Type` |    -    | Description of the param. |

**Returns**

| Type         | Description                |
|--------------|----------------------------|
| `ReturnType` | Description of the return. |

**Raises**

| Exception | Description               |
|-----------|---------------------------|
| `Error`   | Description of the raise. |
