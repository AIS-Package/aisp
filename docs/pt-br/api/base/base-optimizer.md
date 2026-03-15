---
id: base-optimizer
sidebar_label: BaseOptimizer
keywords:
  - base
  - otimizar
  - otimização
  - otimizar interface
  - objective function
  - minimization
  - maximization
tags:
  - otimizar
  - otimização
---

# BaseOptimizer

Classe base abstrata para algoritmos de otimização.

> **Módulos:** `aisp.base`  
> **Importação:** `from aisp.base import BaseOptimizer`

---

## Visão geral

Esta classe define a interface principal para algoritmos de otimização.  
Ela mantém o histórico de custos, soluções avaliadas, e a melhor solução encontrada durante a otimização. As classes
derivadas devem implementar os métodos ``optimize`` e ``affinity_function``.

Casos de uso:

- Classe base abstrata para estender classes de algoritmos de otimização.

---

## Atributos

| Nome               | Tipo              | Padrão  | Descrição                                                     |
|--------------------|-------------------|:-------:|---------------------------------------------------------------|
| `cost_history`     | `List[float]`     |  `[]`   | Histórico dos melhores custos encontrados em cada iteração.   |
| `solution_history` | `List`            |  `[]`   | Histórico da melhor solução encontrada em cada iteração.      |
| `best_solution`    | `Any`             | `None`  | A melhor solução global encontrada.                           |
| `best_cost`        | `Optional[float]` | `None`  | Custo da melhor solução global encontrada.                    |
| `mode`             | `{"min", "max"}`  | `'min'` | Define se o algoritmo minimiza ou maximiza a função de custo. |

---

## Métodos abstratos

### optimize

```python
@abstractmethod
def optimize(
    self,
    max_iters: int = 50,
    n_iter_no_change: int = 10,
    verbose: bool = True
) -> Any:
    ...
```

Executa o processo de otimização.  
Este método abstrato é implementado é responsabilidade das classes filhas, definindo a estratégia de otimização.

**Parâmetros**

| Nome               | Tipo   | Padrão | Descrição                                                            |
|--------------------|--------|:------:|----------------------------------------------------------------------|
| `max_iters`        | `int`  |  `50`  | Número máximo de iterações                                           |
| `n_iter_no_change` | `int`  |  `10`  | Número máximo de interações sem atualização da melhor solução.       |
| `verbose`          | `bool` | `True` | Indica se as mensagens de progresso do treinamento deve ser exibido. |

**Returns**

``BaseClassifier`` - Retorna a instância da classe.


---

### affinity_function

```python
@abstractmethod
def affinity_function(self, solution: Any) -> float:
    ...
```

Avalia a afinidade (qualidade) de uma solução candidata.

Este método deve ser implementado conforme o problema de otimização específico, definindo como a solução sera medida.
O valor retornado deve representar a qualidade da solução avaliada.

**Parâmetros**

| Nome       | Tipo  | Padrão | Descrição                            |
|------------|-------|:------:|--------------------------------------|
| `solution` | `Any` |   -    | Solução candidata que será avaliada. |

**Returns**

`float` - Valor de custo associada a solução encontrada.

---

## Métodos públicos

### get_report

```python
def get_report(self) -> str:
    ...
```

Gera um relatorio resumindo e formatado do processo de otimização.  
O relatorio incluir a melhor solução, seu custo, e a evolução dos valores a cada iteração.

**Returns**

`str` - Uma string formatada contendo o resumo da otimização.

---

### register

```python
def register(self, alias: str, function: Callable[..., Any]) -> None:
    ...
```

Registra dinamicamente uma função na instância do otimizador.

**Parâmetros**

| Nome       | Tipo                 | Padrão | Descrição                                          |
|------------|----------------------|:------:|----------------------------------------------------|
| `alias`    | `str`                |   -    | Nome usado para acessar a função como um atributo. |
| `function` | `Callable[..., Any]` |   -    | Função que será registrada.                        |

**Exceções**

`TypeError` - Lançado quando não é uma função valida.

`AttributeError` - Lançado quando o `alias` esta protegido e não pode ser modificado, ou se não existir na classe.

---

### reset

```python
def reset(self):
    ...
```

Reseta o estado interno do objeto, limpando histórico e restaurando valores iniciais.
