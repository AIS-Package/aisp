---
id: table-formatter
sidebar_label: TableFormatter
---

# TableFormatter

Format tabular data into strings for display in the console.

> **Module:** `aisp.utils.display`  
> **Import:** `from aisp.utils.display import TableFormatter`

---

## Constructor Parameters

| Name      | Type                | Default | Description                                                                                    |
|-----------|---------------------|:-------:|------------------------------------------------------------------------------------------------|
| `headers` | `Mapping[str, int]` |    -    | Mapping of column names to their respective widths, in the format {column_name: column_width}. |

---

## Public Methods

### get_header

```python
def get_header(self):
    ...
```

Generate the table header, including the top border, column headings, and separator line.

**Returns**

`str` - Formatted string of the table header.

---

### get_row

```python
def get_row(self, values: Mapping[str, Union[str, int, float]]):
    ...
```

Generate a formatted row for the table data.

**Parameters**

| Name     | Type                                   | Default | Description                                                                 |
|----------|----------------------------------------|:-------:|-----------------------------------------------------------------------------|
| `values` | `Mapping[str, Union[str, int, float]]` |    -    | Dictionary with values for each column, in the format {column_name: value}. |

**Returns**

`str` - Formatted string of the table row.

---

### get_bottom

```python
def get_bottom(self, new_line: bool = False):
    ...
```

Generate the table's bottom border.

**Parameters**

| Name       | Type   | Default | Description                                                      |
|------------|--------|:-------:|------------------------------------------------------------------|
| `new_line` | `bool` | `False` | If True, adds a line break before the border (default is False). |

**Returns**

`str` - Formatted string for the bottom border.
