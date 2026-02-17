"""Artificial Immune Network for Optimization (Opt-AiNet)."""

from typing import Any, Optional, Callable, Literal, Dict

import numpy as np
import numpy.typing as npt

from ..base import BaseOptimizer
from ..utils.random import set_seed_numba
from ..utils.sanitizers import sanitize_param, sanitize_seed, sanitize_bounds
from ..utils.types import FeatureTypeAll


class OptAiNet(BaseOptimizer):
    """Artificial Immune Network for Optimization.

    Parameters
    ----------
    problem_size : int
        Dimension of the problem to be minimized.
    N : int, default=50
        Number of memory cells (antibodies) in the population.
    rate_clonal : float, default=10
        Maximum number of possible clones of a cell. This value is multiplied by
        cell_affinity to determine the number of clones.
    n_diversity_injection : int, default=5
        Number of new random memory cells injected to maintain diversity.
    affinity_function : Optional[Callable[..., npt.NDArray]], default=None
        Objective function to evaluate candidate solutions in minimizing the problem.
            feature_type : FeatureTypeAll, default='ranged-features'
        Type of problem samples: binary, continuous, or based on value ranges.
        Specifies the type of features: "continuous-features", "binary-features",
        "ranged-features", or "permutation-features".
    bounds : Optional[Dict], default=None
        Definition of search limits when ``feature_type='ranged-features'``.
        Can be provided in two ways:

        * Fixed values: ``{'low': float, 'high': float}``
            Values are replicated across all dimensions, generating equal limits for each
            dimension.
        * Arrays: ``{'low': list, 'high': list}``
            Each dimension has specific limits. Both arrays must be
            ``problem_size``.

    mode : Literal["min", "max"], default="min"
        Defines whether the algorithm minimizes or maximizes the cost function.
    seed : Optional[int], default=None
        Seed for random generation of detector values. If None, the value is random.
    """

    def __init__(
        self,
        problem_size: int,
        N: int = 50,
        rate_clonal: int = 10,
        n_diversity_injection: int = 5,
        affinity_function: Optional[Callable[..., npt.NDArray]] = None,
        feature_type: FeatureTypeAll = 'ranged-features',
        bounds: Optional[Dict] = None,
        mode: Literal["min", "max"] = "min",
        seed: Optional[int] = None
    ):
        super().__init__(affinity_function)
        self.problem_size = sanitize_param(problem_size, 1, lambda x: x > 0)
        self.N: int = sanitize_param(N, 50, lambda x: x > 0)
        self.rate_clonal: int = sanitize_param(rate_clonal, 10, lambda x: x > 0)
        self.n_diversity_injection: int = sanitize_param(
            n_diversity_injection, 5, lambda x: x > 0
        )

        self.feature_type: FeatureTypeAll = feature_type

        self._bounds: Optional[Dict] = None
        self._bounds_extend_cache: Optional[np.ndarray] = None
        self.bounds = bounds

        self.mode: Literal["min", "max"] = sanitize_param(
            mode,
            "min",
            lambda x: x == "max"
        )

        self.seed: Optional[int] = sanitize_seed(seed)
        if self.seed is not None:
            np.random.seed(self.seed)
            set_seed_numba(self.seed)

    @property
    def bounds(self) -> Optional[Dict]:
        """Getter for the bounds attribute."""
        return self._bounds

    @bounds.setter
    def bounds(self, value: Optional[Dict]):
        """Setter for the bounds attribute."""
        if self.feature_type == 'ranged-features':
            self._bounds = sanitize_bounds(value, self.problem_size)
            low_bounds = np.array(self._bounds['low'])
            high_bounds = np.array(self._bounds['high'])
            self._bounds_extend_cache = np.array([low_bounds, high_bounds])
        else:
            self._bounds = None
            self._bounds_extend_cache = None

    def optimize(self, max_iters: int = 50, n_iter_no_change=10, verbose: bool = True) -> Any:
        """Execute the optimization process and return the population.

        Parameters
        ----------
        max_iters : int, default=50
            Maximum number of interactions when searching for the best solution using clonalg.
        n_iter_no_change: int, default=10
            the maximum number of iterations without updating the best cell
        verbose : bool, default=True
            Feedback on interactions, indicating the best antibody.

        Returns
        -------
        population : any
        """
        return []
