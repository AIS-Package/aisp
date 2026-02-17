"""Base class for optimization algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, List, Any, Callable

from ._base import Base
from ...utils.display import TableFormatter


class BaseOptimizer(ABC, Base):
    """Abstract base class for optimization algorithms.

    This class defines the core interface for optimization strategies. It keeps track of cost
    history, evaluated solutions, and the best solution found during the optimization process.
    Subclasses must implement ``optimize`` and ``affinity_function``.

    Parameters
    ----------
    affinity_function : Optional[Callable[..., Any]], default=None
        Objective function to evaluate candidate solutions the problem.

    Attributes
    ----------
    cost_history : List[float]
        History of best costs found at each iteration.
    solution_history : List
        History of the best solution found at each iteration.
    best_solution : Any
        The best solution found.
    best_cost : Optional[float]
        Cost of the best solution found.
    mode : {"min", "max"}, default="min"
        Defines whether the algorithm minimizes or maximizes the cost function.
    """

    def __init__(self, affinity_function: Optional[Callable[..., Any]] = None) -> None:
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
        self.mode = "min"
        if callable(affinity_function):
            self.register('affinity_function', affinity_function)

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
        best_solution : Any
            The best solution associated with the given cost.
        """
        self._solution_history.append(best_solution)
        self._cost_history.append(cost)
        is_better = (
            self._best_cost is None or
            (self.mode == "min" and cost < self._best_cost) or
            (self.mode == "max" and cost > self._best_cost)
        )
        if is_better:
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
            f"Best cost      : {self.best_cost}\n",
            f"Best solution  : {self.best_solution}\n",
            "Cost History per Iteration:\n"
        ]
        table_formatter = TableFormatter(
            {
                'Iteration': 12,
                'Cost': 28
            }
        )

        report_parts.extend([table_formatter.get_header()])

        for i, cost in enumerate(self._cost_history, start=1):
            report_parts.append(
                '\n' + table_formatter.get_row(
                    {
                        'Iteration': f"{i:>11} ",
                        'Cost': f"{cost:>27.6f} "
                    }
                )
            )

        report_parts.append(table_formatter.get_bottom(True))
        return "".join(report_parts)

    @abstractmethod
    def optimize(
        self,
        max_iters: int = 50,
        n_iter_no_change: int = 10,
        verbose: bool = True
    ) -> Any:
        """Execute the optimization process.

        This abstract method must be implemented by the subclass, defining
        how the optimization strategy explores the search space.

        Parameters
        ----------
        max_iters : int, default=50
            Maximum number of iterations
        n_iter_no_change : int, default=10
            the maximum number of iterations without updating the best
        verbose : bool, default=True
            Flag to enable or disable detailed output during optimization.

        Returns
        -------
        best_solution : Any
            The best solution found by the optimization algorithm.
        """

    def affinity_function(self, solution: Any) -> float:
        """Evaluate the affinity of a candidate solution.

        This abstract method must be implemented by the subclass to define the problem-specific.

        Parameters
        ----------
        solution : Any
            Candidate solution to be evaluated.

        Returns
        -------
        affinity : float
            Cost value associated with the given solution.

        Raises
        ------
        NotImplementedError
            If no affinity function has been provided.
        """
        raise NotImplementedError(
            "No affinity function to evaluate the candidate cell was provided."
        )

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
        if alias in self._protected_aliases or alias.startswith("_"):
            raise AttributeError(f"The alias '{alias}' is protected and cannot be modified.")
        if not hasattr(self, alias):
            raise AttributeError(
                f"Alias '{alias}' is not a valid method of {self.__class__.__name__}"
            )
        setattr(self, alias, function)

    def reset(self):
        """Reset the object's internal state, clearing history and resetting values."""
        self._cost_history = []
        self._solution_history = []
        self._best_solution = None
        self._best_cost = None

    def _affinity_function(self, solution: Any) -> float:
        """
        Evaluate the affinity of a candidate cell.

        Parameters
        ----------
        solution : npt.NDArray
            Candidate solution to evaluate.

        Returns
        -------
        affinity : np.float64
            Affinity value associated with the given cell.

        Raises
        ------
        NotImplementedError
            If no affinity function has been provided.
        """
        return float(self.affinity_function(solution))
