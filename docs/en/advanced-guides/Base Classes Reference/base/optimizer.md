Base class for optimization algorithms.

# BaseOptimizer

This class defines the core interface for optimization strategies and
keeps track of the cost history, evaluated solutions, and the best solution found. Subclasses must implement
``optimize`` and ``objective_function``.

---

### Properties

#### `cost_history`

```python
@property
def cost_history(self) -> List[float]
```

Returns the history of costs during optimization.

---

#### `solution_history`

```python
@property
def solution_history(self) -> List
```

Returns the history of evaluated solutions.

---

#### `best_solution`

```python
@property
def best_solution(self) -> Optional[Any]
```

Returns the best solution found so far, or `None` if unavailable.

---

#### `best_cost`

```python
@property
def best_cost(self) -> Optional[float]
```

Returns the cost of the best solution found so far, or `None` if unavailable.

---

## Functions

### Function _record_best(...)

```python
def _record_best(self, cost: float, best_solution: Any) -> None
```

Record a new cost value and update the best solution if improved.

**Parameters**:
  * ***cost***: `float` - Cost value to be added to the history.

---

### Function get_report()

```python
def get_report(self) -> str
```

Generate a formatted summary report of the optimization process. The report includes the best solution,
its associated cost, and the evolution of cost values per iteration.

**Returns**:
  * **report**: `str` - A formatted string containing the optimization summary.

---

### Function register(...)

```python
def register(self, alias: str, function: Callable[..., Any]) -> None
```

Register a function dynamically in the optimizer instance.

**Parameters**:
  * ***alias***: `str` - Name used to access the function as an attribute.
  * ***function***: `Callable[..., Any]` - Callable to be registered.

**Raises**:
  * **TypeError**: If `function` is not callable.
  * **AttributeError**: If `alias` is protected and cannot be modified, or if `alias` does not exist in the
     optimizer class.

---

### Function reset()

```python
def reset(self)
```

Reset the object's internal state, clearing history and resetting values.

---

### Abstract methods

#### Function optimize(...)

```python
def optimize(self, max_iters: int = 50, n_iter_no_change=10, verbose: bool = True) -> Any
```

Execute the optimization process. This method must be implemented by the subclass to define how the optimization strategy explores the search space.

**Parameters**:
  * ***max_iters***: `int` - Maximum number of iterations.
  * ***n_iter_no_change***: `int`, default=10 - The maximum number of iterations without updating the best solution.
  * ***verbose***: `bool`, default=True - Flag to enable or disable detailed output during optimization.

**Returns**:
  * **best_solution**: `Any` - The best solution found by the optimization algorithm.

---

#### Function affinity_function(...)

```python
def affinity_function(self, solution: Any) -> float
```

Evaluate the affinity of a candidate solution. This method must be implemented by the subclass to define the problem-specific.

**Parameters**:
  * ***solution***: `Any` - Candidate solution to be evaluated.

**Returns**:
  * **cost**: `float` - Cost value associated with the given solution.
