"""Artificial Immune Network (AiNet)."""
from collections import Counter
from heapq import nlargest
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import squareform, pdist
from tqdm import tqdm

from ._base import BaseAiNet
from ..base.mutation import clone_and_mutate_binary, clone_and_mutate_continuous
from ..utils.sanitizers import sanitize_choice, sanitize_param, sanitize_seed
from ..utils.distance import hamming, compute_metric_distance, get_metric_code
from ..utils.types import FeatureType, MetricType
from ..utils.validation import detect_vector_data_type


class AiNet(BaseAiNet):
    """Artificial Immune Network for Compression and Clustering .

    This class implements the aiNet algorithm, an artificial immune network model designed for
    clustering and data compression tasks. The aiNet algorithm uses principles from immune
    network theory, clonal selection, and affinity maturation to compress high-dimensional
    datasets. [1]_
    For clustering, the class uses SciPy’s implementation of the **Minimum Spanning Tree**
    (MST) to remove the most distant nodes and separate the groups. [2]_

    Parameters
    ----------
    N : int, default=50
        Number of memory cells (antibodies) in the population.
    n_clone : int, default=10
        Number of clones generated for each selected memory cell.
    top_clonal_memory_size : Optional[int], default=5
       Number of highest-affinity antibodies selected per antigen for cloning and mutation.
       If set to None or 0, all antibodies are cloned, following the original aiNet algorithm.
    n_diversity_injection : int, default=5
        Number of new random memory cells injected to maintain diversity.
    affinity_threshold : float, default=0.5
        Threshold for affinity (similarity) to determine cell suppression or selection.
    suppression_threshold : float, default=0.5
        Threshold for suppressing similar memory cells.
    mst_pruning_threshold : float, default=0.9
        Fraction of the maximum **Minimum Spanning Tree** (MST) edge weight used as a pruning
        threshold to disconnect clusters in the antibody network.
    max_iterations : int, default=10
        Maximum number of training iterations.
    k : int, default=3
        The number of K nearest neighbors that will be used to choose a label in the prediction.
    metric : Literal["manhattan", "minkowski", "euclidean"], default="euclidean"
        Way to calculate the distance between the detector and the sample:

        * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression:
            √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).

        * ``'minkowski'`` ➜ The calculation of the distance is given by the expression:
            ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.

        * ``'manhattan'`` ➜ The calculation of the distance is given by the expression:
            ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|).

    seed : Optional[int]
        Seed for the random generation of detector values. Defaults to None.
    use_mst_clustering : bool, default=True
        If ``True``, performs clustering with **Minimum Spanning Tree** (MST). If ``False``,
        does not perform clustering and predict returns None.
    **kwargs
        p : float
            This parameter stores the value of ``p`` used in the Minkowski distance. The default
            is ``2``, which represents normalized Euclidean distance.\
            Different values of p lead to different variants of the Minkowski Distance.

    References
    ----------
    .. [1] de Castro, L. N., & Von Zuben, F. J. (2001).
           *aiNet: An Artificial Immune Network for Data Analysis*.
           Draft Chapter XII of the book *Data Mining: A Heuristic Approach*.
           Department of Computer and Automation Engineering, University of Campinas.
           Available at:
             https://www.dca.fee.unicamp.br/~vonzuben/research/lnunes_dout/
             artigos/DMHA.PDF
    .. [2] SciPy Documentation. *Minimum Spanning Tree*.
           https://docs.scipy.org/doc/scipy/reference/generated/
           scipy.sparse.csgraph.minimum_spanning_tree
    """

    def __init__(
        self,
        N: int = 50,
        n_clone: int = 10,
        top_clonal_memory_size: int = 5,
        n_diversity_injection: int = 5,
        affinity_threshold: float = 0.5,
        suppression_threshold: float = 0.5,
        mst_pruning_threshold: float = 0.90,
        max_iterations: int = 10,
        k: int = 3,
        metric: MetricType = "euclidean",
        seed: Optional[int] = None,
        use_mst_clustering: bool = True,
        **kwargs
    ):
        self.N: int = sanitize_param(N, 50, lambda x: x > 0)
        self.n_clone: int = sanitize_param(n_clone, 10, lambda x: x > 0)
        self.top_clonal_memory_size: int = sanitize_param(
            top_clonal_memory_size, 5, lambda x: x > 0
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
        self.mst_pruning_threshold: float = sanitize_param(
            mst_pruning_threshold, 0.99, lambda x: x > 0
        )
        self.max_iterations: int = sanitize_param(max_iterations, 100, lambda x: x > 0)
        self.k: int = sanitize_param(k, 3, lambda x: x > 3)
        self.seed: Optional[int] = sanitize_seed(seed)
        self.use_mst_clustering: bool = use_mst_clustering
        if self.seed is not None:
            np.random.seed(self.seed)

        self._feature_type: FeatureType = "continuous-features"

        self.metric: str = sanitize_choice(
            metric, ["manhattan", "minkowski"], "euclidean"
        )

        self.p: np.float64 = np.float64(kwargs.get("p", 2.0))

        self.classes = []
        self._memory_network: dict = {}
        self._population_antibodies: Optional[npt.NDArray] = None
        self._n_features: int = 0

    @property
    def memory_network(self) -> dict:
        """Return the immune network representing clusters or graph structure."""
        return self._memory_network

    @property
    def population_antibodies(self) -> Optional[npt.NDArray]:
        """Return the set of memory antibodies."""
        return self._population_antibodies

    def fit(self, X: npt.NDArray, verbose: bool = True):
        """
        Train the model using the input data X and corresponding labels y.

        Parameters
        ----------
        X : npt.NDArray
            Input data used for training the model.
        verbose : bool, default=True
            Feedback from the progress bar showing current training interaction details.

        Returns
        -------
        self : AiNet
            Returns the instance of the class that implements this method.
        """
        progress = None

        self._feature_type = detect_vector_data_type(X)

        super()._check_and_raise_exceptions_fit(X, self._feature_type)

        if self._feature_type == "binary-features":
            X = X.astype(np.bool_)
            self.metric = "hamming"

        self._n_features = X.shape[1]

        if verbose:
            progress = tqdm(
                total=self.max_iterations,
                postfix="\n",
                bar_format="{desc} ┇{bar}┇ {n}/{total} total training interactions",
            )

        population_p = self._init_population_antibodies()

        t: int = 1
        while t <= self.max_iterations:
            pool_memory = []
            permutations = np.random.permutation(X.shape[0])
            for antigen in X[permutations]:
                clonal_memory = self._select_and_clone_population(antigen, population_p)
                pool_memory.extend(self._clonal_suppression(antigen, clonal_memory))
            pool_memory = self._memory_suppression(pool_memory)
            t += 1
            if t == self.max_iterations:
                pool_memory.extend(self._diversity_introduction())
            population_p = np.asarray(pool_memory)

            if verbose and progress is not None:
                progress.update(1)
        self._population_antibodies = population_p
        if self.use_mst_clustering:
            self._separate_clusters_by_mst()
        if verbose and progress is not None:
            progress.set_description(
                f"\033[92m✔ Set of memory antibodies for classes "
                f"({', '.join(map(str, self.classes))}) successfully generated | "
                f"Clusters: {len(self.classes)} | Population of antibodies size: "
                f"{len(self._population_antibodies)}\033[0m"
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
        if not self.use_mst_clustering or self._memory_network is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, self._n_features, self._feature_type
        )

        c: list = []

        all_cells_memory = [
            (class_name, cell)
            for class_name in self.classes
            for cell in self._memory_network[class_name]
        ]

        for line in X:
            label_stim_list = [
                (class_name, self._affinity(memory, line))
                for class_name, memory in all_cells_memory
            ]
            # Create the list with the k nearest neighbors and select the class with the most votes
            k_nearest = nlargest(self.k, label_stim_list, key=lambda x: x[1])
            votes = Counter(label for label, _ in k_nearest)
            c.append(votes.most_common(1)[0][0])
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
            self._feature_type
        )

    def _select_and_clone_population(self, antigen: npt.NDArray, population: npt.NDArray) -> list:
        """
        We select the best antibodies and apply hypermutation.

        Parameters
        ----------
        antigen : npt.NDArray
            The antigen for which affinities will be calculated.
        population: list
            The list of antibodies (solutions) to be evaluated and cloned.

        Returns
        -------
        list
            mutated clones
        """
        affinities = np.asarray([self._affinity(antigen, antibody) for antibody in population])

        if self.top_clonal_memory_size is not None and self.top_clonal_memory_size > 0:
            selected_idxs = np.argsort(-affinities)[:self.top_clonal_memory_size]
        else:
            selected_idxs = np.arange(affinities.shape[0])

        clonal_m = []
        for i in selected_idxs:
            clones = self._clone_and_mutate(
                population[i],
                int(self.n_clone * affinities[i])
            )
            clonal_m.extend(clones)

        return clonal_m

    def _clonal_suppression(self, antigen: npt.NDArray, clones: list):
        """
        Suppresses redundant clones based on affinity thresholds.

        This function removes clones whose affinity with the antigen is lower than the defined
        threshold (affinity_threshold) and eliminates redundant clones whose similarity with the
        clones already selected exceeds the suppression threshold (suppression_threshold).

        Parameters
        ----------
        antigen : npt.NDArray
            The antigen for which affinities will be calculated.
        clones : list
            The list of candidate clones to be suppressed.

        Returns
        -------
        list
            A list of non-redundant clones after suppression.
        """
        suppression_affinity = [
            clone for clone in clones
            if self._affinity(clone, antigen) > self.affinity_threshold
        ]
        suppressed_clones = []
        for candidate in suppression_affinity:
            is_redundant = False
            for existing in suppressed_clones:
                if self._affinity(candidate, existing) > self.suppression_threshold:
                    is_redundant = True
                    break
            if not is_redundant:
                suppressed_clones.append(candidate)

        return suppressed_clones

    def _memory_suppression(self, pool_memory: list) -> list:
        """
        Remove redundant antibodies whose similarity exceeds the suppression threshold.

        Calculate the affinity between all memory antibodies and remove redundant antibodies
        whose similarity exceeds the suppression threshold.

        Parameters
        ----------
        pool_memory : list
            antibodies memory.

        Returns
        -------
        list
            Updated memory, without redundancies.
        """
        suppressed_memory = []

        for antibody in pool_memory:
            is_similar = False
            for existing in suppressed_memory:
                if self._affinity(antibody, existing) > self.suppression_threshold:
                    is_similar = True
                    break
            if not is_similar:
                suppressed_memory.append(antibody)

        return suppressed_memory

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
            self._feature_type
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
        distance: float
        if self._feature_type == "binary-features":
            distance = hamming(u, v)
        else:
            distance = compute_metric_distance(
                u, v, get_metric_code(self.metric), self.p
            )

        return 1 - (distance / (1 + distance))

    def _clone_and_mutate(self, antibody: npt.NDArray, n_clone: int) -> npt.NDArray:
        """
        Generate mutated clones from an antigen, based on the data type (binary or continuous).

        The number of clones generated is defined by `n_clone`, and each clone is a modified version
        of the original `antibody` vector. The mutation method applied depends on the sample type.

        Parameters
        ----------
        antibody : npt.NDArray
            Original vector (antigen) from which clones will be generated.
        n_clone : int
            Number of clones to generate.

        Returns
        -------
        npt.NDArray
            Array of the form (n_clone, len(antibody)) containing the mutated clones of the
            original vector.
        """
        if self._feature_type == "binary-features":
            return clone_and_mutate_binary(antibody, n_clone)
        return clone_and_mutate_continuous(antibody, n_clone)

    def _separate_clusters_by_mst(self):
        """Clusters the antibodies using the Minimum Spanning Tree (MST).

        Builds a Minimum Spanning Tree (MST) based on the distance matrix
        between the antibodies in the population. Then, it removes the edges
        whose weight exceeds a threshold proportional to the maximum weight
        in the MST. Each resulting component represents a cluster.
        """
        if self._population_antibodies is None or len(self._population_antibodies) == 0:
            raise ValueError("Population of antibodies is empty")

        kwargs = {'p': self.p} if self.metric == 'minkowski' else {}
        antibodies_matrix = squareform(
            pdist(self._population_antibodies, metric=self.metric, **kwargs)
        )

        antibodies_mst = minimum_spanning_tree(csgraph=antibodies_matrix).toarray()
        threshold = self.mst_pruning_threshold * antibodies_mst.max()

        antibodies_mst[antibodies_mst >= threshold] = 0
        n_antibodies, labels = connected_components(csgraph=antibodies_mst, directed=False)

        self._memory_network = {
            label: self._population_antibodies[labels == label]
            for label in range(n_antibodies)
        }
        self.classes = self._memory_network.keys()
