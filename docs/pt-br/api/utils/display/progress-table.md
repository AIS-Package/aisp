---
id: progress-table
sidebar_label: ProgressTable
---

# ProgressTable

Exibe uma tabela formatada no console para acompanhar o progresso do algoritmo.

:::tip[Herança]

Esta classe herda de [TableFormatter](./table-formatter.md).

:::


> **Módulo:** `aisp.utils.display`  
> **Importação:** `from aisp.utils.display import ProgressTable`

---

## Parâmetros do Construtor

| Nome      | Tipo                | Padrão | Descrição                                         |
|-----------|---------------------|:------:|---------------------------------------------------|
| `headers` | `Mapping[str, int]` |   -    | Mapeamento `{nome_da_coluna: largura_da_coluna}`. |
| `verbose` | `bool`              | `True` | Se False, não imprime nada no terminal.           |

---

## Métodos Públicos

### update

```python
def update(self, values: Mapping[str, Union[str, int, float]]) -> None:
    ...
```

Adiciona uma nova linha de valores à tabela.

**Parâmetros**

| Nome     | Tipo                                   | Padrão | Descrição                                                     |
|----------|----------------------------------------|:------:|---------------------------------------------------------------|
| `values` | `Mapping[str, Union[str, int, float]]` |   -    | As chaves devem corresponder às colunas definidas em headers. |

---

### finish

```python
def finish(self) -> None:
    ...
```

Finaliza a exibição da tabela, imprimindo a borda inferior e o tempo total.
