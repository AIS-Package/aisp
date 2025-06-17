"""Artificial Immune Network."""

from typing import Optional, Literal

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ._base import BaseAiNet
from ..base.mutation import clone_and_mutate_binary, clone_and_mutate_continuous
from ..utils.sanitizers import sanitize_choice, sanitize_param, sanitize_seed
from ..utils.distance import hamming, compute_metric_distance, get_metric_code


class AiNet(BaseAiNet):
    """Artificial Immune Network.

    Parameters
    ----------
    N : int, default=50
        Number of memory cells (antibodies) in the population.
    n_clone : int, default=10
        Number of clones generated for each selected memory cell.
    n_clonal_memory_size : int, default=5
        Size of the clonal memory for each class.
    n_diversity_injection : int, default=5
        Number of new random memory cells injected to maintain diversity.
    affinity_threshold : float, default=0.5
        Threshold for affinity (similarity) to determine cell suppression or selection.
    suppression_threshold : float, default=0.5
        Threshold for suppressing similar memory cells.
    max_iters : int, default=10
        Maximum number of training iterations.
    metric : Literal["manhattan", "minkowski", "euclidean"], default="euclidean"
        Way to calculate the distance between the detector and the sample:

        * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression:
            √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).

        * ``'minkowski'`` ➜ The calculation of the distance is given by the expression:
            ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.

        * ``'manhattan'`` ➜ The calculation of the distance is given by the expression:
            ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|).

    algorithm : {"continuous-features", "binary-features"}, default="continuous-features"
        Specifies the type of algorithm to use based on the nature of the input features:

        * ``continuous-features``: selects an algorithm designed for continuous data, which should
            be normalized within the range [0, 1].

        * ``binary-features``: selects an algorithm specialized for handling binary variables.

    seed : Optional[int]
        Seed for the random generation of detector values. Defaults to None.
    **kwargs
        p : float
            This parameter stores the value of ``p`` used in the Minkowski distance. The default
            is ``2``, which represents normalized Euclidean distance.\
            Different values of p lead to different variants of the Minkowski Distance.
    """

    def __init__(
        self,
        N: int = 50,
        n_clone: int = 10,
        n_clonal_memory_size: int = 5,
        n_diversity_injection: int = 5,
        affinity_threshold: float = 0.5,
        suppression_threshold: float = 0.5,
        max_iters: int = 10,
        metric: Literal["manhattan", "minkowski", "euclidean"] = "euclidean",
        algorithm: Literal[
            "continuous-features", "binary-features"
        ] = "continuous-features",
        seed: Optional[int] = None,
        **kwargs
    ):
        self.N: int = sanitize_param(N, 50, lambda x: x > 0)
        self.n_clone: int = sanitize_param(n_clone, 10, lambda x: x > 0)
        self.n_clonal_memory_size: int = sanitize_param(
            n_clonal_memory_size, 5, lambda x: x > 0
        )
        self.n_diversity_injection: int = sanitize_param(
            n_diversity_injection, 5, lambda x: x > 0
        )
        self.affinity_threshold: float = sanitize_param(
            affinity_threshold, 0.5, lambda x: x > 0
        )
        self.suppression_threshold: float = sanitize_param(
            suppression_threshold, 0.5, lambda x: x > 0
        )
        self.max_iters: int = sanitize_param(max_iters, 100, lambda x: x > 0)
        self.seed: Optional[int] = sanitize_seed(seed)
        if self.seed is not None:
            np.random.seed(self.seed)

        self.algorithm: Literal["continuous-features", "binary-features"] = (
            sanitize_param(
                algorithm, "continuous-features", lambda x: x == "binary-features"
            )
        )

        if algorithm == "binary-features":
            self.metric: str = "hamming"
        else:
            self.metric: str = sanitize_choice(
                metric, ["manhattan", "minkowski"], "euclidean"
            )

        self.p: np.float64 = np.float64(kwargs.get("p", 2.0))

        self.classes = []
        self._memory_network: dict = {}
        self._n_features: int = 0

    @property
    def memory_network(self):
        """Return the set of memory antibodies."""
        return self._memory_network

    def fit(self, X: npt.NDArray, verbose: bool = True):
        """
        Train the model using the input data X and corresponding labels y.

        Parameters
        ----------
        X : npt.NDArray
            Input data used for training the model.
        verbose : bool, default=True
            Flag to enable or disable detailed output during training.

        Returns
        -------
        self : AiNet
            Returns the instance of the class that implements this method.
        """
        progress = None

        super()._check_and_raise_exceptions_fit(X, self.algorithm)

        if self.algorithm == "binary-features":
            X = X.astype(np.bool_)

        self._n_features = X.shape[1]

        if verbose:
            progress = tqdm(
                total=self.max_iters,
                postfix="\n",
                bar_format="{desc} ┇{bar}┇ {n}/{total} total training interactions",
            )

        population_p = self._init_population_antibodies()
        memory = []

        t: int = 1
        while t <= self.max_iters:
            clonal_memory = []
            permutations = np.random.permutation(X.shape[0])
            for antigen in X[permutations]:
                affinities = [self._affinity(antigen, antibody) for antibody in population_p]
                best_idxs = np.argsort(affinities)[:self.n_clonal_memory_size]
                # Gera clones proporcionalmente à afinidade
                for i in best_idxs:
                    clones = self._clone_and_mutate(
                        population_p[i],
                        int(self.n_clone * (1.0 / (1 + affinities[i])))
                    )
                    clonal_memory.extend(clones)

                # Supressão clonal
                clonal_memory = [
                    clone for clone in clonal_memory
                    if self.suppression_threshold < self._affinity(
                        clone, antigen
                    ) < self.affinity_threshold
                ]
                memory.extend(clonal_memory)

            print(clonal_memory)
            memory.extend(self._diversity_introduction())
            population_p = np.array(memory)
            t += 1
            if verbose:
                progress.update(1)
        self._memory_network = memory
        if verbose:
            progress.set_description(
                f"\033[92m✔ Set of memory antibodies for classes "
                f"({', '.join(map(str, self.classes))}) successfully generated\033[0m"
            )
        return self

    def predict(self, X) -> Optional[npt.NDArray]:
        """
        Generate predictions based on the input data X.

        Parameters
        ----------
        X : npt.NDArray
            Input data for which predictions will be generated.

        Returns
        -------
        Predictions : Optional[npt.NDArray]
            Predicted values for each input sample, or ``None`` if the prediction fails.
        """

        if self._memory_network is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, len(self._memory_network[self.classes[0]][0]), self.algorithm
        )

        c: list = []

        return np.array(c)

    def _init_population_antibodies(self) -> npt.NDArray:
        """
        Initialize the antibody set of the network population randomly.

        Returns
        -------
        npt.NDArray
            List of initialized memories.
        """

        return self._generate_random_antibodies(
            self.N,
            self._n_features,
            self.algorithm
        )

    def _diversity_introduction(self):
        """
       Introduce diversity into the antibody population.

       Returns
       -------
       npt.NDArray
           Array of new random antibodies for diversity introduction.
       """

        return self._generate_random_antibodies(
            self.n_diversity_injection,
            self._n_features,
            self.algorithm
        )

    def _affinity(self, u: npt.NDArray, v: npt.NDArray) -> float:
        """
        Calculate the stimulus between two vectors using metrics.

        Parameters
        ----------
        u : npt.NDArray
            Coordinates of the first point.
        v : npt.NDArray
            Coordinates of the second point.

        Returns
        -------
        float
            The stimulus rate between the vectors.
        """
        if self.algorithm == "binary-features":
            return hamming(u, v)
        return compute_metric_distance(u, v, get_metric_code(self.metric), self.p)

    def _clone_and_mutate(self, antibody: npt.NDArray, n_clone: int):
        if self.algorithm == "continuous-features":
            return clone_and_mutate_continuous(antibody, n_clone)

        return clone_and_mutate_binary(antibody, n_clone)
