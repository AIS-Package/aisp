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

## Parâmetros do Construtor

| Nome          | Tipo   | Padrão | Descrição                                              |
|---------------|--------|:------:|--------------------------------------------------------|
| `total`       | `int`  |   -    | Número total de interações.                            |
| `suffix`      | `str`  |   -    | Texto opcional exibido apos a barra de progresso.      |
| `description` | `str`  |   -    | Texto opcional exibido antes a barra de progresso.     |
| `slots`       | `int`  |   -    | Numero de caracteres para formar a barra de progresso. |
| `verbose`     | `bool` |   -    | Se False, não imprime nada no terminal.                |

**Exceções**

| Exceção      | Descrição                                            |
|--------------|------------------------------------------------------|
| `ValueError` | Se `total` ou `slots` forem menores ou igual a zero. |

---

## Métodos Públicos

### set_description

```python
def set_description(self, description: str) -> None:
    ...
```

Atualize o texto da descrição antes da barra de progresso.

**Parâmetros**

| Nome          | Tipo  | Padrão | Description    |
|---------------|-------|:------:|----------------|
| `description` | `str` |   -    | Nova descrição |

### update

```python
def update(self, increment: int = 1) -> None:
    ...
```

Atualiza a barra de progresso, incrementando seu valor pela quantidade informada.

**Parâmetros**

| Nome        | Tipo  | Padrão | Descrição                                                         |
|-------------|-------|:------:|-------------------------------------------------------------------|
| `increment` | `int` |  `1`   | Número de iterações concluídas para adicionar ao progresso atual. |


**Exceções**

| Exceção      | Descrição                      |
|--------------|--------------------------------|
| `ValueError` | Se o `increment` for negativo. |

### finish

```python
def finish(self) -> None:
    ...
```

Encerre a exibição da barra de progresso e imprima o tempo total.


