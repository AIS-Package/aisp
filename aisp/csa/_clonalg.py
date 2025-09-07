"""Clonal Selection Algorithm (CLONALG)."""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np
import numpy.typing as npt

from ..base import BaseOptimizer, set_seed_numba
from ..utils.sanitizers import sanitize_seed


class Clonalg(BaseOptimizer):
    """Clonal Selection Algorithm (CLONALG).

    The Clonal Selection Algorithm (CSA) is an optimization algorithm inspired by the biological
    process of selection and clonal expansion of B and T cells in the immune system [1]_.


    Parameters
    ----------
    affinity_function : Optional[Callable[..., npt.NDArray]], default=None
        Objective function to evaluate candidate solutions in minimizing the problem.
    seed : Optional[int], default=None
        Seed for the random generation of detector values. Defaults to None.

    References
    ----------
    .. [1] BROWNLEE, Jason. Clonal Selection Algorithm. Clever Algorithms: Nature-inspired
    Programming Recipes., 2011. Available at:
    https://cleveralgorithms.com/nature-inspired/immune/clonal_selection_algorithm.html
    """

    def __init__(
        self,
        affinity_function: Optional[Callable[..., npt.NDArray]] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        self._affinity_function = affinity_function
        self.seed: Optional[int] = sanitize_seed(seed)
        if self.seed is not None:
            np.random.seed(self.seed)
            set_seed_numba(self.seed)

    def optimize(self, max_iters: int = 50, verbose: bool = True) -> npt.NDArray:
        """Execute the optimization process and return the best solution.
        Parameters
        ----------
        max_iters : int, default=100
            Maximum number of interactions when searching for the best solution using clonalg.
        verbose : bool, default=True
            Feedback on interactions, indicating the best antibody.
        """
        pass

    def affinity_function(self, cell: npt.NDArray) -> float:
        """
        Evaluate the affinity of a candidate cell.

        Parameters
        ----------
        cell : npt.NDArray
            Candidate solution to evaluate.

        Returns
        -------
        affinity : float
            Affinity value associated with the given cell.

        Raises
        ------
        NotImplementedError
            If no affinity function has been provided.
        """
        if not callable(self._affinity_function):
            raise NotImplementedError(
                "No objective function to evaluate the candidate cell was provided."
            )
        return self._affinity_function(cell)
