---
id: table-formatter
sidebar_label: TableFormatter
---

# TableFormatter

Formata dados em tabelas para exibição no console.

> **Módulo:** `aisp.utils.display`  
> **Importação:** `from aisp.utils.display import TableFormatter`

---

## Parâmetros do Construtor

| Nome      | Tipo                | Padrão | Descrição                                                                                                          |
|-----------|---------------------|:------:|--------------------------------------------------------------------------------------------------------------------|
| `headers` | `Mapping[str, int]` |   -    | Mapeamento dos nomes das colunas para suas respectivas larguras, no formato `{nome_da_coluna: largura_da_coluna}`. |

---

## Métodos Públicos

### get_header

```python
def get_header(self):
    ...
```

Gera o cabeçalho da tabela, incluindo a borda superior, os nomes das colunas e a linha separadora.

**Returns**

| Tipo  | Descrição                             |
|-------|---------------------------------------|
| `str` | String formatada do header da tabela. |

---

### get_row

```python
def get_row(self, values: Mapping[str, Union[str, int, float]]):
    ...
```

Gera uma linha formatada para os dados da tabela.

**Parâmetros**

| Nome     | Tipo                                   | Padrão | Descrição                                                                      |
|----------|----------------------------------------|:------:|--------------------------------------------------------------------------------|
| `values` | `Mapping[str, Union[str, int, float]]` |   -    | Dicionário com valores para cada coluna, no formato `{nome_da_coluna: valor}`. |

**Returns**

| Tipo  | Descrição                            |
|-------|--------------------------------------|
| `str` | String formatada da linha da tabela. |

---

### get_bottom

```python
def get_bottom(self, new_line: bool = False):
    ...
```

Gera a borda inferior da tabela.

**Parâmetros**

| Nome       | Tipo   | Padrão  | Descrição                                                                  |
|------------|--------|:-------:|----------------------------------------------------------------------------|
| `new_line` | `bool` | `False` | Se `True`, adiciona uma quebra de linha antes da borda (padrão é `False`). |

**Returns**

| Tipo  | Descrição                           |
|-------|-------------------------------------|
| `str` | String formatada da borda inferior. |
