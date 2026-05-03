"""Artificial Immune Network for Optimization (Opt-AiNet)."""
import heapq
from typing import Any, Optional, Callable, Literal, Dict, List

import numpy as np
import numpy.typing as npt

from ..base import BaseOptimizer
from ..base.immune.cell import Antibody
from ..base.immune.populations import generate_random_antibodies
from ..utils.display import ProgressTable
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
        self.population = None
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

        self._pop_avg = None

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

    def optimize(
        self,
        max_iters: int = 50,
        n_iter_no_change=10,
        verbose: bool = True
    ) -> List[Antibody]:
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

        self.population = [
            Antibody(cell, self._affinity_function(cell))
            for cell in self._init_population_antibodies()
        ]
        t = 1
        best_cost = None
        stop = 0
        progress = ProgressTable(
            {
                "Iteration": 11,
                f"Best Affinity ({self.mode})": 25,
                "Worse Affinity": 20,
                "Pop Size": 10,
                "Stagnation": 17,
                "avg": 10,
            },
            verbose,
        )
        while t <= max_iters:
            best_antibody = self._select_top_antibody()
            best_cost = best_antibody.affinity
            self._record_best(best_antibody[0].affinity, best_antibody[0].vector)

            self._pop_avg = self._average_affinity(self.population)

            if stop == n_iter_no_change:
                break

            t += 1

        progress.finish()
        return self.population

    def _init_population_antibodies(self) -> npt.NDArray:
        """Initialize the antibody set of the population randomly.

        Returns
        -------
        antibodies : npt.NDArray
            List of initialized antibodies.
        """
        return generate_random_antibodies(
            self.N,
            self.problem_size,
            self.feature_type,
            self._bounds_extend_cache
        )

    def _diversity_introduction(self):
        """Introduce diversity into the antibody population.

        Returns
        -------
        new_antibodies : npt.NDArray
            Array of new random antibodies for diversity introduction.
        """
        return generate_random_antibodies(
            self.n_diversity_injection,
            self.problem_size,
            self.feature_type,
            self._bounds_extend_cache,
        )

    def _select_top_antibody(
        self,
    ) -> Antibody:
        """Select the antibodies with the highest or lowest values, depending on the mode.

        Returns
        -------
        selected : list[Antibody]
            List containing the `n` antibodies selected according to the defined min
            or max criterion.
        """
        if self.mode == "max":
            return heapq.nlargest(1, self.population)[0]

        return heapq.nsmallest(1, self.population)[0]

    @staticmethod
    def _average_affinity(population: list[Antibody]) -> float:
        return np.fromiter((p.affinity for p in population), dtype=float).mean()
