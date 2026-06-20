---
id: progress-bar
sidebar_label: ProgressBar
---

# ProgressBar

Display a console progress bar to track an algorithm's progress.

This class builds a progress bar in the format `{description}{bar*slots}{suffix}` and, when calling the
`finish` method, print the final progress bar along with the total time elapsed since the class was instance.


> **Module:** `aisp.utils.display`  
> **Import:** `from aisp.utils.display import ProgressBar`

---

## Constructor Parameters

| Name          | Type   | Default | Description                                          |
|---------------|--------|:-------:|------------------------------------------------------|
| `total`       | `int`  |    -    | Total number of iterations.                          |
| `suffix`      | `str`  |  `''`   | Optional text displayed after the progress bar.      |
| `description` | `str`  |  `''`   | Optional text displayed before the progress bar.     |
| `slots`       | `int`  |  `10`   | Number of character slots used to represent the bar. |
| `verbose`     | `bool` | `True`  | If False, prints nothing to the terminal.            |

**Raises**

| Exception    | Description                                              |
|--------------|----------------------------------------------------------|
| `ValueError` | If the `total` or `slots` is less than or equal to zero. |

---

## Public Methods

### set_description

```python
def set_description(self, description: str) -> None:
    ...
```

Update the description before the progress bar.

**Parameters**

| Name          | Type  | Default | Description      |
|---------------|-------|:-------:|------------------|
| `description` | `str` |    -    | New description. |

### update

```python
def update(self, increment: int = 1) -> None:
    ...
```

Increment the progress bar.

**Parameters**

| Name        | Type  | Default | Description                                                    |
|-------------|-------|:-------:|----------------------------------------------------------------|
| `increment` | `int` |   `1`   | Number of completed iterations to add to the current progress. |


**Raises**

| Exception    | Description                   |
|--------------|-------------------------------|
| `ValueError` | If the increment is negative. |

### finish

```python
def finish(self) -> None:
    ...
```

End the progress display and print the total elapsed time.


