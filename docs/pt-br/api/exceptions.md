---
id: exceptions
sidebar_label: aisp.exceptions
keywords:
  - exceptions
  - Exceções
  - warnings
---

# aisp.exceptions

Avisos e erros personalizados.

> **Módulo:** `aisp.exceptions`  
> **Importação:** `from aisp import exceptions`

## Classes de exceção

### MaxDiscardsReachedError

```python
class MaxDiscardsReachedError(Exception):
    ...
```

Exceção lançada quando o número máximo de descartes do detector é atingido.

**Parâmetros**

| Nome          | Tipo            | Padrão | Descrição                                |
|---------------|-----------------|:------:|------------------------------------------|
| `object_name` | `str`           |   -    | O nome da classe que lança a exceção.    |
| `message`     | `Optional[str]` | `None` | Mensagem personalizada que será exibida. |

---

### FeatureDimensionMismatch

```python
class FeatureDimensionMismatch(Exception):
    ...
```

Exceção lançada quando o número de características (dimensões) de entrada não corresponde ao esperado.
Essa exceção é acionada durante a predição se a quantidade de características estiver incorreta.

**Parâmetros**

| Nome            | Tipo            | Padrão | Descrição                                   |
|-----------------|-----------------|:------:|---------------------------------------------|
| `expected`      | `int`           |   -    | O número esperado de características.       |
| `received`      | `int`           |   -    | O número real de características recebidas. |
| `variable_name` | `Optional[str]` | `None` | O nome da variável que causou o error.      |

---

### UnsupportedTypeError

```python
class UnsupportedTypeError(Exception):
    ...
```

Exceção lançada quando o tipo de dados do vetor de entrada não é suportado.
Essa exceção é lançada quando o tipo de dado do vetor não corresponde a nenhum dos suportados.

**Parâmetros**

| Nome      | Tipo            | Padrão | Descrição                                |
|-----------|-----------------|:------:|------------------------------------------|
| `message` | `Optional[str]` | `None` | Mensagem personalizada que será exibida. |

---

### ModelNotFittedError

```python
class ModelNotFittedError(Exception):
    ...
```

Exceção lançada quando o método é chamado antes que o modelo ter sido treinado.
Essa exceção ocorre quando a instância do modelo é utilizada sem que ele tenha sido previamente treinado
por meio do método `fit`.

**Parâmetros**

| Nome          | Tipo            | Padrão | Descrição                                |
|---------------|-----------------|:------:|------------------------------------------|
| `object_name` | `str`           |   -    | O nome da classe que lança a exceção.    |
| `message`     | `Optional[str]` | `None` | Mensagem personalizada que será exibida. |
