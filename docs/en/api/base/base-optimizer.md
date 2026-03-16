---
id: base-optimizer
sidebar_label: BaseOptimizer
keywords:
    - base
    - optimizer
    - optimization
    - optimizer interface
    - objective function
    - minimization
    - maximization
tags:
    - optimizer
    - optimization
---

# BaseOptimizer

Abstract base class for optimization algorithms.

> **Module:** `aisp.base`  
> **Import:** `from aisp.base import BaseOptimizer`

---

## Overview

This class defines the core interface for optimization strategies. It keeps track of cost history, evaluated
solutions, and the best solution found during the optimization process. Subclasses must implement ``optimize``
and ``affinity_function``.

Use cases:

- Abstract base class for extending optimization algorithm classes.

---

## Attributes

| Name               | Type              | Default | Description                                                             |
|--------------------|-------------------|:-------:|-------------------------------------------------------------------------|
| `cost_history`     | `List[float]`     |  `[]`   | History of best costs found at each iteration.                          |
| `solution_history` | `List`            |  `[]`   | History of the best solution found at each iteration.                   |
| `best_solution`    | `Any`             | `None`  | The best solution found.                                                |
| `best_cost`        | `Optional[float]` | `None`  | Cost of the best solution found.                                        |
| `mode`             | `{"min", "max"}`  | `'min'` | Defines whether the algorithm minimizes or maximizes the cost function. |

---

## Abstract Methods

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

Execute the optimization process.  
This abstract method must be implemented by the subclass, defining how the optimization strategy explores the search
space.

**Parameters**

| Name               | Type   | Default | Description                                                |
|--------------------|--------|:-------:|------------------------------------------------------------|
| `max_iters`        | `int`  |  `50`   | Maximum number of iterations                               |
| `n_iter_no_change` | `int`  |  `10`   | the maximum number of iterations without updating the best |
| `verbose`          | `bool` | `True`  | Flag to enable or disable detailed output during training. |

**Returns**

``BaseClassifier`` - Returns the instance of the class that implements this method.


---

### affinity_function

```python
@abstractmethod
def affinity_function(self, solution: Any) -> float:
    ...
```

Evaluate the affinity of a candidate solution.

This method must be implemented according to the specific optimization problem, defining how the solution will be
evaluated.
The returned value should represent the quality of the evaluated solution.

**Parameters**

| Name       | Type  | Default | Description                         |
|------------|-------|:-------:|-------------------------------------|
| `solution` | `Any` |    -    | Candidate solution to be evaluated. |

**Returns**

`float` - Cost value associated with the given solution.

---

## Public Methods

### get_report

```python
def get_report(self) -> str:
    ...
```

Generate a formatted summary report of the optimization process.  
The report includes the best solution, its associated cost, and the evolution of cost values per iteration.

**Returns**

`str` - A formatted string containing the optimization summary.

---

### register

```python
def register(self, alias: str, function: Callable[..., Any]) -> None:
    ...
```

Register a function dynamically in the optimizer instance.

**Parameters**

| Name       | Type                 | Default | Description                                       |
|------------|----------------------|:-------:|---------------------------------------------------|
| `alias`    | `str`                |    -    | Name used to access the function as an attribute. |
| `function` | `Callable[..., Any]` |    -    | Callable to be registered.                        |

**Raises**

`TypeError` - If `function` is not callable.

`AttributeError` - If `alias` is protected and cannot be modified. Or if `alias` does not exist in the optimizer class.

---

### reset

```python
def reset(self):
    ...
```

Reset the object's internal state, clearing history and resetting values.
