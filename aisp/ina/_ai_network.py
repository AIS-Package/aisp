"""Artificial Immune Network (AiNet)."""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.spatial.distance import squareform, pdist, cdist
from tqdm import tqdm

from ..base import BaseClusterer
from ..base.immune.cell import Cell
from ..base.immune.mutation import (
    clone_and_mutate_binary,
    clone_and_mutate_continuous,
    clone_and_mutate_ranged,
)
from ..base.immune.populations import generate_random_antibodies
from ..utils.distance import hamming, compute_metric_distance, get_metric_code
from ..utils.multiclass import predict_knn_affinity
from ..utils.random import set_seed_numba
from ..utils.sanitizers import sanitize_choice, sanitize_param, sanitize_seed
from ..utils.types import FeatureType, MetricType
from ..utils.validation import (
    detect_vector_data_type,
    check_array_type,
    check_feature_dimension,
    check_binary_array,
)


class AiNet(BaseClusterer):
    """Artificial Immune Network for Compression and Clustering.

    This class implements the aiNet algorithm, an artificial immune network model designed for
    clustering and data compression tasks. The aiNet algorithm uses principles from immune
    network theory, clonal selection, and affinity maturation to compress high-dimensional
    datasets. [1]_
    For clustering, the class uses SciPy's implementation of the **Minimum Spanning Tree**
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
    mst_inconsistency_factor : float, default=2.0
        Factor used to determine which edges in the **Minimum Spanning Tree (MST)**
        are considered inconsistent.
    max_iterations : int, default=10
        Maximum number of training iterations.
    k : int, default=3
        The number of K nearest neighbors that will be used to choose a label in the prediction.
    metric : Literal["manhattan", "minkowski", "euclidean"], default="euclidean"
        Way to calculate the distance between the detector and the sample:

        * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression:
            √( (x₁ - x₂)² + (y₁ - y₂)² + ... + (yn - yn)²).

        * ``'minkowski'`` ➜ The calculation of the distance is given by the expression:
            ( |X₁ - Y₁|p + |X₂ - Y₂|p + ... + |Xn - Yn|p) ¹/ₚ.

        * ``'manhattan'`` ➜ The calculation of the distance is given by the expression:
            ( |x₁ - x₂| + |y₁ - y₂| + ... + |yn - yn|).

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
    .. [1] De Castro, Leandro & José, Fernando & von Zuben, Antonio Augusto. (2001). aiNet: An
           Artificial Immune Network for Data Analysis.
           Available at:
             https://www.researchgate.net/publication/
             228378350_aiNet_An_Artificial_Immune_Network_for_Data_Analysis
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
        mst_inconsistency_factor: float = 2.0,
        max_iterations: int = 10,
        k: int = 3,
        metric: MetricType = "euclidean",
        seed: Optional[int] = None,
        use_mst_clustering: bool = True,
        **kwargs,
    ):
        self.N: int = sanitize_param(N, 50, lambda x: x > 0)
        self.n_clone: int = sanitize_param(n_clone, 10, lambda x: x > 0)

        self.top_clonal_memory_size: Optional[int] = None
        if top_clonal_memory_size is not None:
            self.top_clonal_memory_size = sanitize_param(
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
        self.mst_inconsistency_factor: float = sanitize_param(
            mst_inconsistency_factor, 2, lambda x: x >= 0
        )
        self.max_iterations: int = sanitize_param(max_iterations, 10, lambda x: x > 0)
        self.k: int = sanitize_param(k, 1, lambda x: x > 0)
        self.seed: Optional[int] = sanitize_seed(seed)
        self.use_mst_clustering: bool = use_mst_clustering
        if self.seed is not None:
            np.random.seed(self.seed)
            set_seed_numba(self.seed)

        self._feature_type: FeatureType = "continuous-features"
        self.metric: str = sanitize_choice(
            metric, ["euclidean", "manhattan", "minkowski"], "euclidean"
        )

        self.p: np.float64 = np.float64(kwargs.get("p", 2.0))
        self._metric_params = {}
        if self.metric == "minkowski":
            self._metric_params["p"] = self.p

        self.classes: Optional[npt.NDArray] = None
        self._memory_network: Dict[int, List[Cell]] = {}
        self._population_antibodies: Optional[npt.NDArray] = None
        self._n_features: int = 0
        self._bounds: Optional[npt.NDArray[np.float64]] = None
        self._mst_structure: Optional[npt.NDArray] = None
        self._mst_mean_distance: Optional[float] = None
        self._mst_std_distance: Optional[float] = None
        self._all_cells_memory_vectors: Optional[
            List[Tuple[str | int, npt.NDArray]]
        ] = None

    @property
    def memory_network(self) -> Dict[int, List[Cell]]:
        """Return the immune network representing clusters or graph structure."""
        return self._memory_network

    @property
    def population_antibodies(self) -> Optional[npt.NDArray]:
        """Return the set of memory antibodies."""
        return self._population_antibodies

    @property
    def mst(self) -> dict:
        """Returns the Minimum Spanning Tree and its statistics."""
        return {
            "graph": self._mst_structure,
            "mean_distance": self._mst_mean_distance,
            "std_distance": self._mst_std_distance,
        }

    def fit(self, X: npt.NDArray, verbose: bool = True) -> AiNet:
        """
        Train the AiNet model on input data.

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
        self._feature_type = detect_vector_data_type(X)

        check_array_type(X)

        match self._feature_type:
            case "binary-features":
                X = X.astype(np.bool_)
                self.metric = "hamming"
            case "ranged-features":
                self._bounds = np.vstack([np.min(X, axis=0), np.max(X, axis=0)])

        self._n_features = X.shape[1]

        progress = tqdm(
            total=self.max_iterations,
            postfix="\n",
            disable=not verbose,
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

            if t < self.max_iterations:
                pool_memory.extend(self._diversity_introduction())
            population_p = np.asarray(pool_memory)

            progress.update()

            t += 1
        self._population_antibodies = population_p

        if self.use_mst_clustering:
            self._build_mst()
            self.update_clusters()
            labels = self.classes.tolist() if self.classes is not None else []
            progress.set_description(
                f"\033[92m✔ Set of memory antibodies for classes "
                f"({', '.join(map(str, labels))}) successfully generated | "
                f"Clusters: {len(labels)} | Population of antibodies size: "
                f"{len(self._population_antibodies)}\033[0m"
            )
        else:
            progress.set_description(
                f"\033[92m✔ Set of memory antibodies successfully generated | "
                f"Population of antibodies size: {len(self._population_antibodies)}\033[0m"
            )
        progress.close()

        return self

    def predict(self, X) -> Optional[np.ndarray]:
        """
        Predict cluster labels for input data.

        Parameters
        ----------
        X : npt.NDArray
            Data to predict.

        Returns
        -------
        Predictions : Optional[npt.NDArray]
            Predicted cluster labels, or None if clustering is disabled.
        """
        if (
            not self.use_mst_clustering
            or self._memory_network is None
            or self._all_cells_memory_vectors is None
        ):
            return None

        check_feature_dimension(X, self._n_features)
        if self._feature_type == "binary-features":
            check_binary_array(X)

        return predict_knn_affinity(
            X, self.k, self._all_cells_memory_vectors, self._affinity
        )

    def _init_population_antibodies(self) -> npt.NDArray:
        """
        Initialize the antibody set of the network population randomly.

        Returns
        -------
        npt.NDArray
            List of initialized memories.
        """
        return generate_random_antibodies(
            self.N, self._n_features, self._feature_type, self._bounds
        )

    def _select_and_clone_population(
        self, antigen: npt.NDArray, population: npt.NDArray
    ) -> list:
        """
        Select top antibodies by affinity and generate mutated clones.

        Parameters
        ----------
        antigen : npt.NDArray
            The antigen for which affinities will be calculated.
        population: list
            The list of antibodies (solutions) to be evaluated and cloned.

        Returns
        -------
        list[npt.NDArray]
            List of mutated clones.
        """
        affinities = self._calculate_affinities(antigen, population)

        if self.top_clonal_memory_size is not None and self.top_clonal_memory_size > 0:
            selected_idxs = np.argsort(-affinities)[: self.top_clonal_memory_size]
        else:
            selected_idxs = np.arange(affinities.shape[0])

        clonal_m: list = []
        for i in selected_idxs:
            clones = self._clone_and_mutate(
                population[i], int(self.n_clone * affinities[i])
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
            Non-redundant, high-affinity clones.
        """
        suppression_affinity = [
            clone
            for clone in clones
            if self._affinity(clone, antigen) > self.affinity_threshold
        ]
        return self._memory_suppression(suppression_affinity)

    def _memory_suppression(self, pool_memory: list) -> list:
        """
        Remove redundant antibodies from memory pool.

        Calculate the affinity between all memory antibodies and remove redundant antibodies
        whose similarity exceeds the suppression threshold.

        Parameters
        ----------
        pool_memory : list
            antibodies memory.

        Returns
        -------
        list
            Memory pool without redundant antibodies.
        """
        if not pool_memory:
            return []
        suppressed_memory = [pool_memory[0]]
        for candidate in pool_memory[1:]:
            affinities = self._calculate_affinities(
                candidate.reshape(1, -1), np.asarray(suppressed_memory)
            )

            if not np.any(affinities > self.suppression_threshold):
                suppressed_memory.append(candidate)
        return suppressed_memory

    def _diversity_introduction(self):
        """
        Introduce diversity into the antibody population.

        Returns
        -------
        npt.NDArray
            Array of new random antibodies for diversity introduction.
        """
        return generate_random_antibodies(
            self.n_diversity_injection,
            self._n_features,
            self._feature_type,
            self._bounds,
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
            Affinity score in [0, 1], where higher means more similar.
        """
        distance: float
        if self._feature_type == "binary-features":
            distance = hamming(u, v)
        else:
            distance = compute_metric_distance(
                u, v, get_metric_code(self.metric), self.p
            )

        return float(1 - (distance / (1 + distance)))

    def _calculate_affinities(self, u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
        """
        Calculate the affinity matrix between a reference vector and a set of target vectors.

        Parameters
        ----------
        u : npt.NDArray
            An array with shape (n_features).
        v : npt.NDArray
            An array of vectors with shape (n_samples, n_features).


        Returns
        -------
        npt.NDArray
            One-dimensional array of shape (n_samples,), containing the affinities between `u`
            and each vector in `v`.
        """
        u = np.reshape(u, (1, -1))
        v = np.atleast_2d(v)
        distances = cdist(u, v, metric=self.metric, **self._metric_params)[0]  # type: ignore

        return 1 - (distances / (1 + distances))

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
        if self._feature_type == "binary-features":
            return clone_and_mutate_binary(antibody, n_clone)
        if self._feature_type == "ranged-features" and self._bounds is not None:
            return clone_and_mutate_ranged(
                antibody, n_clone, self._bounds, np.float64(1.0)
            )
        return clone_and_mutate_continuous(antibody, n_clone, np.float64(1.0))

    def _build_mst(self):
        """Construct the Minimum Spanning Tree (MST) for the antibody population.

        Computes the pairwise distances between antibodies, builds the MST from
        these distances, and stores the MST structure along with the mean and
        standard deviation of its edge weights.

        Raises
        ------
        ValueError
            If the antibody population is empty.
        """
        if self._population_antibodies is None or len(self._population_antibodies) == 0:
            raise ValueError("Population of antibodies is empty")

        antibodies_matrix = squareform(
            pdist(
                self._population_antibodies, metric=self.metric, **self._metric_params
            )
        )
        antibodies_mst = minimum_spanning_tree(antibodies_matrix).toarray()
        self._mst_structure = antibodies_mst
        nonzero_edges = antibodies_mst[antibodies_mst > 0]
        self._mst_mean_distance = (
            float(np.mean(nonzero_edges)) if nonzero_edges.size else 0.0
        )
        self._mst_std_distance = (
            float(np.std(nonzero_edges)) if nonzero_edges.size else 0.0
        )

    def update_clusters(self, mst_inconsistency_factor: Optional[float] = None):
        """Partition the clusters based on the MST inconsistency factor.

        Uses the precomputed Minimum Spanning Tree (MST) of the antibody population
        to redefine clusters. Edges whose weights exceed the mean plus the
        `mst_inconsistency_factor` multiplied by the standard deviation of MST edge
        weights are removed. Each connected component after pruning is treated as a
        distinct cluster.

        Parameters
        ----------
        mst_inconsistency_factor : float, optional
            If provided, overrides the current inconsistency factor.

        Raises
        ------
        ValueError
            If the Minimum Spanning Tree (MST) has not yet been created
            If Population of antibodies is empty
            If MST statistics (mean or std) are not available.

        Updates
        -------
        self._memory_network : dict[int, npt.NDArray]
            Dictionary mapping cluster labels to antibody arrays.
        self.classes : list
            List of cluster labels.
        """
        if self._mst_structure is None:
            raise ValueError(
                "The Minimum Spanning Tree (MST) has not yet been created."
            )

        if self._population_antibodies is None or len(self._population_antibodies) == 0:
            raise ValueError("Population of antibodies is empty")

        if self._mst_mean_distance is None or self._mst_std_distance is None:
            raise ValueError("MST statistics (mean or std) are not available.")

        if mst_inconsistency_factor is not None:
            self.mst_inconsistency_factor = mst_inconsistency_factor

        antibodies_mst = self._mst_structure.copy()

        thresholds = antibodies_mst > (
            self._mst_mean_distance
            + self.mst_inconsistency_factor * self._mst_std_distance
        )
        antibodies_mst[thresholds] = 0

        n_antibodies, labels = connected_components(
            csgraph=antibodies_mst, directed=False
        )

        self._memory_network = {
            label: [Cell(a) for a in self._population_antibodies[labels == label]]
            for label in range(n_antibodies)
        }
        self.classes = np.array(list(self._memory_network.keys()))

        self._all_cells_memory_vectors = [
            (class_name, cell.vector)
            for class_name in self.classes
            for cell in self._memory_network[class_name]
        ]
