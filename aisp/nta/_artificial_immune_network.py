"""Artificial Immune Network."""
from collections import Counter
from heapq import nlargest
from typing import Optional, Literal

import numpy as np
from numpy import typing as npt
from tqdm import tqdm

from ._base import BaseAiNet
from ..utils.sanitizers import sanitize_param, sanitize_seed, sanitize_choice
from ..utils.distance import hamming, compute_metric_distance, get_metric_code


class AiNet(BaseAiNet):
    """Artificial Immune Network

    Parameters
    ----------
    N : int, default=50
        Number of memory cells (antibodies) in the population.
    k : int, default=3
        Number of nearest neighbors considered for affinity calculation.
    n_clone : int, default=10
        Number of clones generated for each selected memory cell.
    n_clonal_memory_size : int, default=5
        Size of the clonal memory for each class.
    n_diversity_injection : int, default=5
        Number of new random memory cells injected to maintain diversity.
    rate_random_init : float, default=0.5
        Percentage of initial population generated randomly. The remaining (1 - rate_random_init)
        fraction is sampled randomly from the antigen list.
    affinity_threshold : float, default=0.5
        Threshold for affinity (similarity) to determine cell suppression or selection.
    suppression_threshold : float, default=0.5
        Threshold for suppressing similar memory cells.
    max_iters : int, default=100
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
        Specifies the type of features in the input data.
    seed : Literal["continuous-features", "binary-features"], default="continuous-features"
        Specifies the type of algorithm to use based on the nature of the input features:

        * ``continuous-features``: selects an algorithm designed for continuous data, which should
            be normalized within the range [0, 1].

        * ``binary-features``: selects an algorithm specialized for handling binary variables.
        Seed for the random generation of detector values. Defaults to None.
    
    **kwargs
        p : float
            This parameter stores the value of ``p`` used in the Minkowsks distance. The default
            is ``2``, which represents normalized Euclidean distance.\
            Different values of p lead to different variants of the Minkowski Distance.
    """

    def __init__(
        self,
        N: int = 50,
        k: int = 3,
        n_clone: int = 10,
        n_clonal_memory_size: int = 5,
        n_diversity_injection: int = 5,
        rate_random_init: float = 0.5,
        affinity_threshold: float = 0.5,
        suppression_threshold: float = 0.5,
        max_iters: int = 100,
        metric: Literal["manhattan", "minkowski", "euclidean"] = "euclidean",
        algorithm: Literal[
            "continuous-features", "binary-features"
        ] = "continuous-features",
        seed: int = None,
        **kwargs
    ):
        self.N: int = sanitize_param(N, 50, lambda x: x > 0)
        self.k: int = sanitize_param(k, 3, lambda x: x > 3)
        self.n_clone: int = sanitize_param(n_clone, 10, lambda x: x > 0)
        self.n_clonal_memory_size: int = sanitize_param(
            n_clonal_memory_size, 5, lambda x: x > 0
        )
        self.n_diversity_injection: int = sanitize_param(
            n_diversity_injection, 5, lambda x: x > 0
        )
        self.rate_random_init: float = sanitize_param(
            rate_random_init, 0.5, lambda x: x > 0
        )
        self.affinity_threshold: float = sanitize_param(
            affinity_threshold, 0.5, lambda x: x > 0
        )
        self.suppression_threshold: float = sanitize_param(
            suppression_threshold, 0.5, lambda x: x > 0
        )
        self.max_iters: int = sanitize_param(max_iters, 100, lambda x: x > 0)
        self.seed: int = sanitize_seed(seed)
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

        self.classes = None
        self._memory_antibodies: dict = {}

    @property
    def memory_antibodies(self):
        """Return the set of memory antibodies."""
        return self._memory_antibodies

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        Train the model using the input data X and corresponding labels y.

        Parameters
        ----------
        X : npt.NDArray
            Input data used for training the model.
        y : npt.NDArray
            Corresponding labels or target values for the input data.
        verbose : bool, default=True
            Flag to enable or disable detailed output during training.

        Returns
        -------
        self : AiNet
            Returns the instance of the class that implements this method.
        """
        progress = None

        super()._check_and_raise_exceptions_fit(X, y, self.algorithm)

        if self.algorithm == "binary-features":
            X = X.astype(np.bool_)

        self.classes = np.unique(y)
        sample_index = self._slice_index_list_by_class(y)
        if verbose:
            progress = tqdm(
                total=self.max_iters * len(self.classes),
                postfix="\n",
                bar_format="{desc} ┇{bar}┇ {n}/{total} total training interactions",
            )

        for _class_ in self.classes:
            if verbose:
                progress.set_description_str(
                    f"Creating memory antibodies for the {_class_} class:"
                )

            x_class = X[sample_index[_class_]]

            population_p = self._init_population_antibodies(x_class)

            t: int = 1
            while t <= self.max_iters:

                t += 1
                if verbose:
                    progress.update(1)

            self._memory_antibodies[_class_] = population_p

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

        if self._memory_antibodies is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, len(self._memory_antibodies[self.classes[0]][0]), self.algorithm
        )

        c: list = []

        all_memory_antibodies = [
            (class_name, antibody)
            for class_name in self.classes
            for antibody in self._memory_antibodies[class_name]
        ]

        for line in X:
            label_stim_list = [
                (class_name, self._affinity(antibody, line))
                for class_name, antibody in all_memory_antibodies
            ]
            # Create the list with the k nearest neighbors and select the class with the most votes
            k_nearest = nlargest(self.k, label_stim_list, key=lambda x: x[1])
            votes = Counter(label for label, _ in k_nearest)
            c.append(votes.most_common(1)[0][0])
        return np.array(c)

    def _init_population_antibodies(self, antigens_list: npt.NDArray) -> Optional[npt.NDArray]:
        """
        Initializes the antibodies of the network population by randomly selecting a portion
        of the antigens and complementing them with randomly generated vectors. When
        rate_random_init = 1, 100% of the antibodies are generated randomly, without using
        antigens from the training list.

        Parameters
        ----------
        antigens_list : npt.NDArray
            List of training antigens.

        Returns
        -------
        npt.NDArray
            List of initialized memories.
        """
        n, n_features = antigens_list.shape
        if n == 0 or n_features == 0:
            return None

        n_random_init = int(self.N * self.rate_random_init)
        n_antigens_select = self.N - n_random_init

        antigens = None
        if self.rate_random_init < 1:
            indexs = np.random.choice(n, size=n_antigens_select, replace=False)
            antigens = antigens_list[indexs]

        if self.algorithm == "continuous-features":
            random_init = np.random.random_sample(size=(n_random_init, n_features))
        else:
            random_init = np.random.randint(
                0, 2, size=(n_random_init, n_features)
            ).astype(dtype=np.bool_)

        return np.vstack([antigens, random_init]) if antigens else random_init

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
