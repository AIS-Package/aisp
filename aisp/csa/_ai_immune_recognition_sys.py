"""Artificial Immune Recognition System (AIRS)."""

import random
from collections import Counter
from heapq import nlargest
from operator import attrgetter
from typing import List, Optional, Dict

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from tqdm import tqdm


from ._cell import Cell
from ..utils.sanitizers import sanitize_param, sanitize_seed, sanitize_choice
from ..utils.distance import hamming, compute_metric_distance, get_metric_code
from ..utils.types import FeatureType, MetricType
from ..utils.validation import detect_vector_data_type
from ._base import BaseAIRS


class _ARB(Cell):
    """ARB (Artificial recognition ball).

    Individual from the set of recognizing cells (ARB), inherits characteristics from a B-cell,
    adding resource consumption

    Parameters
    ----------
    vector : npt.NDArray
        A vector of cell features.
    stimulation : Optional[float], default=None
        The rate at which the cell stimulates antigens.
    """

    def __init__(
        self,
        vector: npt.NDArray,
        stimulation: Optional[float] = None
    ) -> None:
        super().__init__(vector)
        self.resource: float = 0.0
        if stimulation is not None:
            self.stimulation: float = stimulation

    def consume_resource(self, n_resource: float, amplified: float = 1) -> float:
        """
        Update the amount of resources available for an ARB after consumption.

        This function consumes the resources and returns the remaining amount of resources after
        consumption.

        Parameters
        ----------
        n_resource : float
            Amount of resources.
        amplified : float
            Amplifier for the resource consumption by the cell. It is multiplied by the cell's
            stimulus. The default value is 1.

        Returns
        -------
        n_resource : float
            The remaining amount of resources after consumption.
        """
        consumption = self.stimulation * amplified
        n_resource -= consumption
        if n_resource < 0:
            return 0

        self.resource = consumption
        return n_resource

    def to_cell(self) -> Cell:
        """Convert this _ARB into a pure Cell object."""
        return Cell(self.vector)


class AIRS(BaseAIRS):
    """Artificial Immune Recognition System (AIRS).

    The Artificial Immune Recognition System (AIRS) is a classification algorithm inspired by the
    clonal selection process of the biological immune system. This implementation is based on the
    simplified AIRS2 version described in [1]_. The algorithm has been adapted to support both
    real-valued (continuous) and binary feature datasets.

    Parameters
    ----------
    n_resources : float, default=10
            Total amount of available resources.
    rate_clonal : float, default=10
        Maximum number of possible clones of a class. This quantity is multiplied by (
        cell_stimulus * rate_hypermutation) to define the number of clones.
    rate_mc_init : float, default=0.2
            Percentage of samples used to initialize memory cells.
    rate_hypermutation : float, default=0.75
            The rate of mutated clones derived from rate_clonal as a scalar factor.
    affinity_threshold_scalar : float, default=0.75
            Normalized affinity threshold.
    k : int, default=3
        The number of K nearest neighbors that will be used to choose a label in the prediction.
    max_iters : int, default=100
        Maximum number of interactions in the refinement process of the ARB set exposed to aᵢ.
    resource_amplified : float, default=1.0
        Resource consumption amplifier is multiplied with the incentive to subtract resources.
        Defaults to 1.0 without amplification.
    metric : Literal["manhattan", "minkowski", "euclidean"], default="euclidean"
        Way to calculate the distance between the detector and the sample:

        * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression:
            √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).

        * ``'minkowski'`` ➜ The calculation of the distance is given by the expression:
            ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.

        * ``'manhattan'`` ➜ The calculation of the distance is given by the expression:
            ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|).

    seed : int
        Seed for the random generation of detector values. Defaults to None.

    **kwargs
        p : float
            This parameter stores the value of ``p`` used in the Minkowski distance. The default
            is ``2``, which represents normalized Euclidean distance.\
            Different values of p lead to different variants of the Minkowski Distance.

    Notes
    -----
    This implementation is inspired by AIRS2, a simplified version of the original AIRS algorithm.
    Introducing adaptations to handle continuous and binary datasets.

    Based on Algorithm 16.5 from Brabazon et al. [1]_.

    Related and noteworthy works: access here [2]_.

    References
    ----------
    .. [1] Brabazon, A., O’Neill, M., & McGarraghy, S. (2015). Natural Computing Algorithms. In
        Natural Computing Series. Springer Berlin Heidelberg.
        https://doi.org/10.1007/978-3-662-43631-8

    .. [2] AZZOUG, Aghiles. Artificial Immune Recognition System V2.
        Available at: https://github.com/AghilesAzzoug/Artificial-Immune-System
    """

    def __init__(
        self,
        n_resources: float = 10,
        rate_clonal: int = 10,
        rate_mc_init: float = 0.2,
        rate_hypermutation: float = 0.75,
        affinity_threshold_scalar: float = 0.75,
        k: int = 3,
        max_iters: int = 100,
        resource_amplified: float = 1.0,
        metric: MetricType = "euclidean",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.n_resources: float = sanitize_param(n_resources, 10, lambda x: x >= 1)
        self.rate_mc_init: float = sanitize_param(
            rate_mc_init, 0.2, lambda x: 0 < x <= 1
        )
        self.rate_clonal: int = sanitize_param(rate_clonal, 10, lambda x: x > 0)
        self.rate_hypermutation: float = sanitize_param(
            rate_hypermutation, 0.75, lambda x: x > 0
        )
        self.affinity_threshold_scalar: float = sanitize_param(
            affinity_threshold_scalar, 0.75, lambda x: x > 0
        )
        self.resource_amplified: float = sanitize_param(
            resource_amplified, 1, lambda x: x > 1
        )
        self.k: int = sanitize_param(k, 3, lambda x: x > 3)
        self.max_iters: int = sanitize_param(max_iters, 100, lambda x: x > 0)
        self.seed: Optional[int] = sanitize_seed(seed)
        if self.seed is not None:
            np.random.seed(self.seed)

        self._feature_type: FeatureType = "continuous-features"

        self.metric = sanitize_choice(
            metric, ["manhattan", "minkowski"], "euclidean"
        )

        self.p: np.float64 = np.float64(kwargs.get("p", 2.0))

        self._cells_memory = None
        self.affinity_threshold = 0.0
        self.classes = []
        self._bounds: Optional[npt.NDArray[np.float64]] = None

    @property
    def cells_memory(self) -> Optional[Dict[str, list[Cell]]]:
        """Returns the trained cells memory, organized by class."""
        return self._cells_memory

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True) -> "AIRS":
        """
        Fit the model to the training data using the AIRS.

        The function ``fit(...)``, performs the training according to ``X`` and ``y``, using the
        method AIRS.

        Parameters
        ----------
        X : npt.NDArray
            Training array, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].
        y : npt.NDArray
            Array of target classes of ``X`` with [``N samples`` (lines)].
        verbose : bool
            Feedback on which sample aᵢ the memory cells are being generated.

        Returns
        -------
        AIRS
            Returns the instance itself.
        """
        progress = None

        self._feature_type = detect_vector_data_type(X)

        super()._check_and_raise_exceptions_fit(X, y)

        match self._feature_type:
            case "binary-features":
                X = X.astype(np.bool_)
                self.metric = "hamming"
            case "ranged-features":
                self._bounds = np.vstack([np.min(X, axis=0), np.max(X, axis=0)])

        self.classes = np.unique(y)
        sample_index = self._slice_index_list_by_class(y)
        if verbose:
            progress = tqdm(
                total=len(y),
                postfix="\n",
                bar_format="{desc} ┇{bar}┇ {n}/{total} memory cells for each aᵢ",
            )
        pool_cells_classes = {}
        for _class_ in self.classes:
            if verbose and progress is not None:
                progress.set_description_str(
                    f"Generating the memory cells for the {_class_} class:"
                )

            x_class = X[sample_index[_class_]]
            # Calculating the similarity threshold between antigens
            self._cells_affinity_threshold(x_class)
            sufficiently_similar = (
                self.affinity_threshold * self.affinity_threshold_scalar
            )
            # Initialize memory cells for a class.
            pool_c: list[Cell] = self._init_memory_c(x_class)

            for ai in x_class:
                # Calculating the stimulation of memory cells with aᵢ and selecting the largest
                # stimulation from the memory set.
                c_match = pool_c[0]
                match_stimulation = -1
                for cell in pool_c:
                    stimulation = self._affinity(cell.vector, ai)
                    if stimulation > match_stimulation:
                        match_stimulation = stimulation
                        c_match = cell

                arb_list: list[_ARB] = [
                    _ARB(
                        vector=c_match.vector,
                        stimulation=match_stimulation
                    )
                ]

                set_clones: npt.NDArray = c_match.hyper_clonal_mutate(
                    int(self.rate_hypermutation * self.rate_clonal * match_stimulation),
                    self._feature_type
                )

                for clone in set_clones:
                    arb_list.append(
                        _ARB(
                            vector=clone,
                            stimulation=self._affinity(clone, ai),
                        )
                    )

                c_candidate = self._refinement_arb(ai, match_stimulation, arb_list)

                if c_candidate.stimulation > match_stimulation:
                    pool_c.append(c_candidate.to_cell())
                    if self._affinity(c_candidate.vector, c_match.vector) < sufficiently_similar:
                        pool_c.remove(c_match)

                if verbose and progress is not None:
                    progress.update(1)
            pool_cells_classes[_class_] = pool_c

        if verbose and progress is not None:
            progress.set_description(
                f"\033[92m✔ Set of memory cells for classes ({', '.join(map(str, self.classes))}) "
                f"successfully generated\033[0m"
            )
        self._cells_memory = pool_cells_classes
        return self

    def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]:
        """
        Predict class labels based on the memory cells created during training.

        This method uses the trained memory cells to perform classification of the input data
        using the k-nearest neighbors approach.

        Parameters
        ----------
        X : npt.NDArray
            Array with input samples with [``N samples`` (Lines)] and [``N characteristics``(
            Columns)]

        Returns
        -------
        C : npt.NDArray or None
            An ndarray of the form ``C`` [``N samples``], containing the predicted classes for
            ``X``. or ``None``: If there are no detectors for the prediction.
        """
        if self._cells_memory is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, len(self._cells_memory[self.classes[0]][0].vector), self._feature_type
        )

        c: list = []

        all_cells_memory = [
            (class_name, cell.vector)
            for class_name in self.classes
            for cell in self._cells_memory[class_name]
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

    def _refinement_arb(
        self, ai: npt.NDArray, c_match_stimulation: float, arb_list: List[_ARB]
    ) -> _ARB:
        """
        Refine the ARB set until the average stimulation exceeds the defined threshold.

        This method iteratively refines the ARB set by comparing the average stimulation
        against the `affinity_threshold_scalar`. Refinement continues through multiple iterations
        until the threshold is met or exceeded.

        Parameters
        ----------
        ai : npt.NDArray
            The current antigen.
        c_match_stimulation : float
            The highest stimulation relative to aᵢ
        arb_list : List[_ARB]
            ARB set.

        Returns
        -------
        _ARB
            The cell with the highest ARB stimulation

        Notes
        -----
        Based on Algorithm 16.6 from Brabazon et al. [1]_.

        References
        ----------
        .. [1] Brabazon, A., O’Neill, M., & McGarraghy, S. (2015).
                Natural Computing Algorithms. Natural Computing Series.
                Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-43631-8
        """
        iters = 0
        while True:
            iters += 1
            arb_list.sort(key=attrgetter("stimulation"), reverse=True)
            resource = self.n_resources
            for arb in arb_list:
                resource = arb.consume_resource(
                    n_resource=resource, amplified=self.resource_amplified
                )
                if resource == 0:
                    break
            # remove cells without resources and calculate the average ARB stimulus.
            arb_list = [cell for cell in arb_list if cell.resource > 0]
            if not arb_list:
                break
            avg_stimulation = sum(item.stimulation for item in arb_list) / len(arb_list)

            if iters == self.max_iters or avg_stimulation > self.affinity_threshold:
                break

            # pick a random cell for mutations.
            random_index = random.randint(0, len(arb_list) - 1)
            clone_arb = arb_list[random_index].hyper_clonal_mutate(
                int(self.rate_clonal * c_match_stimulation),
                self._feature_type
            )

            arb_list = [
                _ARB(
                    vector=clone,
                    stimulation=self._affinity(clone, ai)
                )
                for clone in clone_arb
            ]

        return max(arb_list, key=attrgetter("stimulation"))

    def _cells_affinity_threshold(self, antigens_list: npt.NDArray):
        """
        Calculate the affinity threshold based on the average affinity between training instances.

        This function calculates the affinity threshold based on the average affinity between
        training instances, where aᵢ and aⱼ are a pair of antigens, and affinity
        is measured by distance (Euclidean, Manhattan, Minkowski, Hamming).
        Following the formula:

        > affinity_threshold = (Σᵢ=₁ⁿ⁻¹ Σⱼ=ᵢ₊₁ⁿ affinity(aᵢ, aⱼ)) / (n(n-1)/2

        Parameters
        ----------
        antigens_list : npt.NDArray
            List of training antigens.
        """
        if self._feature_type == "binary-features":
            distances = pdist(antigens_list, metric="hamming")
        else:
            metric_kwargs = {'p': self.p} if self.metric == 'minkowski' else {}
            distances = pdist(antigens_list, metric=self.metric, **metric_kwargs)

        n = antigens_list.shape[0]
        sum_affinity = np.sum(1.0 - (distances / (1.0 + distances)))
        self.affinity_threshold = 1.0 - (sum_affinity / ((n * (n - 1)) / 2))

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

    def _init_memory_c(self, antigens_list: npt.NDArray) -> List[Cell]:
        """
        Initialize memory cells by randomly selecting `rate_mc_init` antigens.

        Parameters
        ----------
        antigens_list : npt.NDArray
            List of training antigens.

        Returns
        -------
        List[Cell]
            List of initialized memories.
        """
        n = antigens_list.shape[0]
        n_cells = int(n * self.rate_mc_init)

        if n == 0 or n_cells == 0:
            return []

        permutation = np.random.permutation(n)
        selected = antigens_list[permutation[:n_cells]]
        return [Cell(ai) for ai in selected]
