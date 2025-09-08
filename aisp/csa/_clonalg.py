"""Clonal Selection Algorithm (CLONALG)."""

from __future__ import annotations

import heapq
from typing import Optional, Callable, Dict

import numpy as np
import numpy.typing as npt

from ..base import BaseOptimizer, set_seed_numba
from ..base.mutation import clone_and_mutate_binary, clone_and_mutate_ranged, \
    clone_and_mutate_continuous
from ..base.populations import generate_random_antibodies
from ..utils.sanitizers import sanitize_seed, sanitize_param, sanitize_bounds
from ..utils.types import FeatureType


class Clonalg(BaseOptimizer):
    """Clonal Selection Algorithm (CLONALG).

    The Clonal Selection Algorithm (CSA) is an optimization algorithm inspired by the biological
    process of clonal selection and expansion of antibodies in the immune system [1]_. This
    implementation of CLONALG has been adapted for the minimization of cost functions in binary,
    continuous, and ranged-value problems.


    Parameters
    ----------
    problem_size : int
        Dimension of the problem to be minimized.
    N : int, default=50
        Number of memory cells (antibodies) in the population.
    rate_clonal : float, default=10
        Maximum number of possible clones of a cell. This value is multiplied by
        (cell_stimulus * rate_hypermutation) to determine the number of clones.
    rate_hypermutation : float, default=0.75
        Rate of mutated clones derived from `rate_clonal`, used as a scalar factor.
    n_diversity_injection : int, default=5
        Number of new random memory cells injected to maintain diversity.
    selection_size : int, default=5
        Number of the best antibodies selected for cloning.
    affinity_function : Optional[Callable[..., npt.NDArray]], default=None
        Objective function to evaluate candidate solutions in minimizing the problem.
    feature_type : FeatureType, default='ranged-features'
        Type of problem samples: binary, continuous, or based on value ranges.
        Specifies the type of features: "continuous-features", "binary-features",
        or "ranged-features".
    bounds : Optional[Dict], default=None
        Definition of search limits when ``feature_type='ranged-features'``.
        Can be provided in two ways:

        * Fixed values: ``{'min': float, 'max': float}``
            Values are replicated across all dimensions, generating equal limits for each
            dimension.
        * Arrays: ``{'min': list, 'max': list}``
            Each dimension has specific limits. Both arrays must be
            ``problem_size``.

    seed : Optional[int], default=None
        Seed for random generation of detector values. If None, the value is random.


    References
    ----------
    .. [1] BROWNLEE, Jason. Clonal Selection Algorithm. Clever Algorithms: Nature-inspired
    Programming Recipes., 2011. Available at:
    https://cleveralgorithms.com/nature-inspired/immune/clonal_selection_algorithm.html
    """

    def __init__(
        self,
        problem_size: int,
        N: int = 50,
        rate_clonal: int = 10,
        rate_hypermutation: float = 0.75,
        n_diversity_injection: int = 5,
        selection_size: int = 5,
        affinity_function: Optional[Callable[..., npt.NDArray]] = None,
        feature_type: FeatureType = 'ranged-features',
        bounds: Optional[Dict] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.problem_size = sanitize_param(problem_size, 1, lambda x: x > 0)
        self.N: int = sanitize_param(N, 50, lambda x: x > 0)
        self.rate_clonal: int = sanitize_param(rate_clonal, 10, lambda x: x > 0)
        self.rate_hypermutation: float = sanitize_param(
            rate_hypermutation, 0.75, lambda x: x > 0
        )
        self.n_diversity_injection: int = sanitize_param(
            n_diversity_injection, 5, lambda x: x > 0
        )
        self.selection_size: int = sanitize_param(
            selection_size, 5, lambda x: x > 0
        )
        self._affinity_function = affinity_function
        self.feature_type: FeatureType = feature_type

        self._bounds = None
        self._bounds_extend_cache = None
        self.bounds = bounds

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
            min_bounds = np.array(self._bounds['min'])
            max_bounds = np.array(self._bounds['max'])
            self._bounds_extend_cache = np.array([min_bounds, max_bounds])
        else:
            self._bounds = None
            self._bounds_extend_cache = None

    def optimize(
        self,
        max_iters: int = 50,
        n_iter_no_change=10,
        verbose: bool = True
    ) -> npt.NDArray:
        """Execute the optimization process and return the best solution.

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
        population : npt.NDArray
            Antibody population after clonal expansion.
        """
        population = self._init_population_antibodies()

        t = 1
        antibodies = [(antibody, self.affinity_function(antibody)) for antibody in population]
        best_cost = None
        stop = 0
        while t <= max_iters:
            p_select = heapq.nsmallest(self.selection_size, antibodies, key=lambda x: x[1])
            self._record_best(p_select[0][1], p_select[0][0])

            clones = self._clone_and_hypermutation(p_select)
            p_select.extend(clones)
            p_select = heapq.nsmallest(
                self.N - self.n_diversity_injection,
                p_select,
                key=lambda x: x[1]
            )
            p_rand = [
                (antibody, self.affinity_function(antibody))
                for antibody in self._diversity_introduction()
            ]
            antibodies = p_select + p_rand
            if best_cost == self.best_cost:
                stop += 1
            else:
                stop = 1
                best_cost = self.best_cost

            if stop == n_iter_no_change:
                break

            t += 1

        return np.array([antibody for antibody, _ in antibodies])

    def affinity_function(self, solution: npt.NDArray) -> float:
        """
        Evaluate the affinity of a candidate cell.

        Parameters
        ----------
        solution : npt.NDArray
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
        return float(self._affinity_function(solution))

    def _init_population_antibodies(self) -> npt.NDArray:
        """Initialize the antibody set of the population randomly.

        Returns
        -------
        npt.NDArray
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
        npt.NDArray
            Array of new random antibodies for diversity introduction.
        """
        return generate_random_antibodies(
            self.n_diversity_injection,
            self.problem_size,
            self.feature_type,
            self._bounds_extend_cache
        )

    def _clone_and_mutate(self, antibody: npt.NDArray, n_clone: int) -> npt.NDArray:
        """
        Generate mutated clones from an antibody, based on the feature type.

        Parameters
        ----------
        antibody : npt.NDArray
            Original antibody vector to be cloned and mutated.
        n_clone : int
            Number of clones to generate.

        Returns
        -------
        npt.NDArray
            Array of shape (n_clone, len(antibody)) containing mutated clones
        """
        if self.feature_type == "binary-features":
            return clone_and_mutate_binary(antibody, n_clone)
        if self.feature_type == "ranged-features" and self._bounds_extend_cache is not None:
            return clone_and_mutate_ranged(antibody, n_clone, self._bounds_extend_cache)
        return clone_and_mutate_continuous(antibody, n_clone)

    def _clone_and_hypermutation(
        self,
        population: list[tuple]
    ) -> list:
        """Clone and hypermutate the population's antibodies.

        The clone list is returned with the clones and their affinities with respect to the cost
        function.

        Parameters
        ----------
        population: list
            The list of antibodies (solutions) to be evaluated and cloned.

        Returns
        -------
        list[npt.NDArray]
            List of mutated clones.
        """
        clonal_m = []
        for antibody, affinity in population:
            clones = self._clone_and_mutate(
                antibody,
                abs(int(self.rate_clonal * affinity * self.rate_hypermutation))
            )
            for clone in clones:
                clonal_m.append(
                    (clone, self.affinity_function(clone))
                )

        return clonal_m
