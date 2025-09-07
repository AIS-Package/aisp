"""Base class for optimization algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Any, Callable


class BaseOptimizer(ABC):
    """Abstract base class for optimization algorithms.

    This class defines the core interface for optimization strategies. It keeps track of cost
    history, evaluated solutions, and the best solution found during the optimization process.
    Subclasses must implement ``optimize`` and ``objective_function``.
    """

    def __init__(self) -> None:
        self._cost_history: List[float] = []
        self._solution_history: list = []
        self._best_solution: Optional[Any] = None
        self._best_cost: Optional[float] = None
        self._protected_aliases = [
            "__init__",
            "optimize",
            "register",
            "get_report"
        ]

    @property
    def cost_history(self) -> List[float]:
        """Return the history of costs during optimization."""
        return self._cost_history

    @property
    def solution_history(self) -> List:
        """Returns the history of evaluated solutions."""
        return self._solution_history

    @property
    def best_solution(self) -> Optional[Any]:
        """Return the best solution found so far, or ``None`` if unavailable."""
        return self._best_solution

    @property
    def best_cost(self) -> Optional[float]:
        """Return the cost of the best solution found so far, or ``None`` if unavailable."""
        return self._best_cost

    def _record_best(self, cost: float, best_solution: Any) -> None:
        """Record a new cost value and update the best solution if improved.

        Parameters
        ----------
        cost : float
            Cost value to be added to the history.
        """
        self._solution_history.append(best_solution)
        self._cost_history.append(cost)
        if self._best_cost is None or cost < self._best_cost:
            self._best_solution = best_solution
            self._best_cost = cost

    def get_report(self) -> str:
        """Generate a formatted summary report of the optimization process.

        The report includes the best solution, its associated cost, and the evolution of cost
        values per iteration.

        Returns
        -------
        report : str
            A formatted string containing the optimization summary.
        """
        if not self._cost_history:
            return "Optimization has not been run. The report is empty."

        header = "\n" + "=" * 45 + "\n"
        report_parts = [
            header,
            f"{'Optimization Summary':^45}",
            header,
            f"Best cost      : {self.best_cost:.6f}",
            f"Best solution  : {self.best_solution}\n",
            "Cost History per Iteration:"
        ]

        table_header = f"┃{'Iteration':^12}┃{'Cost':^28}┃"
        table_sep = "┣" + "━" * 12 + "╋" + "━" * 28 + "┫"
        table_footer = "┗" + "━" * 12 + "┻" + "━" * 28 + "┛"
        report_parts.extend([table_header, table_sep])

        for i, cost in enumerate(self._cost_history, start=1):
            report_parts.append(f"┃{i:>11} ┃{cost:>27.6f} ┃")

        report_parts.append(table_footer)
        return "".join(report_parts)

    @abstractmethod
    def optimize(self, *args, verbose: bool = True) -> Any:
        """Execute the optimization process.

        This abstract method must be implemented by the subclass, defining
        how the optimization strategy explores the search space.

        Parameters
        ----------
        *args : tuple
            Variable arguments required by the specific optimization algorithm.
        verbose : bool, default=True
            Flag to enable or disable detailed output during optimization.

        Returns
        -------
        best_solution : Any
            The best solution found by the optimization algorithm.
        """

    @abstractmethod
    def affinity_function(self, solution: Any) -> float:
        """Evaluate the affinity of a candidate solution.

        This abstract method must be implemented by the subclass to define the problem-specific.

        Parameters
        ----------
        solution : Any
            Candidate solution to be evaluated.

        Returns
        -------
        cost : float
            Cost value associated with the given solution.
        """

    def register(self, alias: str, function: Callable[..., Any]) -> None:
        """Register a function dynamically in the optimizer instance.

        Parameters
        ----------
        alias : str
            Name used to access the function as an attribute.
        function : Callable[..., Any]
            Callable to be registered.

        Raises
        ------
        TypeError
            If `function` is not callable.
        AttributeError
            If `alias` is protected and cannot be modified. Or if `alias` does not exist in the
            optimizer class.
        """
        if not callable(function):
            raise TypeError(f"Expected a function for '{alias}', got {type(function).__name__}")
        if alias in self._protected_aliases:
            raise AttributeError(f"The alias '{alias}' is protected and cannot be modified.")
        if not hasattr(self, alias):
            raise AttributeError(
                f"Alias '{alias}' is not a valid method of {self.__class__.__name__}"
            )
        setattr(self, alias, function)
