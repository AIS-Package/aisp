---
id: progress-table
sidebar_label: ProgressTable
---

# ProgressTable

Display a formatted table in the console to track the algorithm's progress.

:::tip[Inheritance]

This class extends [TableFormatter](./table-formatter.md).

:::


> **Module:** `aisp.utils.display`  
> **Import:** `from aisp.utils.display import ProgressTable`

---

## Constructor Parameters

| Name      | Type                | Default | Description                               |
|-----------|---------------------|:-------:|-------------------------------------------|
| `headers` | `Mapping[str, int]` |    -    | Mapping {column_name: column_width}.      |
| `verbose` | `bool`              | `True`  | If False, prints nothing to the terminal. |

---

## Public Methods

### update

````python
def update(self, values: Mapping[str, Union[str, int, float]]) -> None:
    ...
````

Add a new row of values to the table.

**Parameters**

| Name     | Type                                   | Default | Description                                     |
|----------|----------------------------------------|:-------:|-------------------------------------------------|
| `values` | `Mapping[str, Union[str, int, float]]` |    -    | Keys must match the columns defined in headers. |

---

### finish

````python
def finish(self) -> None:
    ...
````

End the table display, printing the bottom border and total time.
