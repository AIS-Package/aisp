"""Artificial Immune Recognition System (AIRS)."""

from __future__ import annotations

import random
from operator import attrgetter
from typing import List, Optional, Dict, Tuple, Any, Union

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from tqdm import tqdm

from ._artificial_recognition_ball import _ARB
from ..base import BaseClassifier
from ..base.immune.cell import BCell
from ..exceptions import ModelNotFittedError
from ..utils.distance import hamming, compute_metric_distance, get_metric_code
from ..utils.multiclass import predict_knn_affinity
from ..utils.random import set_seed_numba
from ..utils.sanitizers import sanitize_param, sanitize_seed, sanitize_choice
from ..utils.types import FeatureType, MetricType
from ..utils.validation import (
    detect_vector_data_type,
    check_array_type,
    check_shape_match,
    check_feature_dimension,
    check_binary_array,
)


class AIRS(BaseClassifier):
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
    metric : {"euclidean", "minkowski", "manhattan"}, default="euclidean"
        Distance metric used to compute affinity between cells and samples.
    seed : int
        Seed for the random generation of detector values. Defaults to None.

    **kwargs
        p : float
            This parameter stores the value of ``p`` used in the Minkowski distance. The default
            is ``2``, which represents normalized Euclidean distance.\
            Different values of p lead to different variants of the Minkowski Distance.

    Attributes
    ----------
    cells_memory : Optional[Dict[str | int, list[BCell]]]
        Dictionary of trained memory cells, organized by class.

    Notes
    -----
    This implementation is inspired by AIRS2, a simplified version of the original AIRS algorithm.
    Introducing adaptations to handle continuous and binary datasets.

    Based on Algorithm 16.5 from Brabazon et al. [1]_.

    Related and noteworthy works: access here [2]_.

    References
    ----------
    .. [1] Brabazon, A., O'Neill, M., & McGarraghy, S. (2015). Natural Computing Algorithms. In
        Natural Computing Series. Springer Berlin Heidelberg.
        https://doi.org/10.1007/978-3-662-43631-8

    .. [2] AZZOUG, Aghiles. Artificial Immune Recognition System V2.
        Available at: https://github.com/AghilesAzzoug/Artificial-Immune-System

    Examples
    --------
    >>> import numpy as np
    >>> from aisp.csa import AIRS

    >>> np.random.seed(1)
    >>> # Generating training data
    >>> a = np.random.uniform(high=0.5, size=(50, 2))
    >>> b = np.random.uniform(low=0.51, size=(50, 2))
    >>> x_train = np.vstack((a, b))
    >>> y_train = [0] * 50 + [1] * 50
    >>> # AIRS Instance
    >>> airs = AIRS(n_resources=5, rate_clonal=5, rate_hypermutation=0.65, seed=1)
    >>> airs = airs.fit(x_train, y_train, verbose=False)
    >>> x_test = [
    ...     [0.15, 0.45],  # Expected: Class 0
    ...     [0.85, 0.65],  # Esperado: Classe 1
    ... ]
    >>> y_pred = airs.predict(x_test)
    >>> print(y_pred)
    [0 1]
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
        self.k: int = sanitize_param(k, 3, lambda x: x > 0)
        self.max_iters: int = sanitize_param(max_iters, 100, lambda x: x > 0)
        self.seed: Optional[int] = sanitize_seed(seed)
        if self.seed is not None:
            np.random.seed(self.seed)
            set_seed_numba(self.seed)

        self._feature_type: FeatureType = "continuous-features"

        self.metric = sanitize_choice(metric, ["manhattan", "minkowski"], "euclidean")

        self.p: np.float64 = np.float64(kwargs.get("p", 2.0))

        self._cells_memory: Optional[Dict[str | int, list[BCell]]] = None
        self._all_class_cell_vectors: Optional[List[Tuple[Any, np.ndarray]]] = None
        self.affinity_threshold: float = 0.0
        self.classes: Optional[npt.NDArray] = None
        self._bounds: Optional[npt.NDArray[np.float64]] = None

    @property
    def cells_memory(self) -> Optional[Dict[str | int, list[BCell]]]:
        """Returns the trained cells memory, organized by class."""
        return self._cells_memory

    def fit(
        self,
        X: Union[npt.NDArray, list],
        y: Union[npt.NDArray, list],
        verbose: bool = True,
    ) -> AIRS:
        """
        Fit the model to the training data using the AIRS.

        The function ``fit(...)``, performs the training according to ``X`` and ``y``, using the
        method AIRS.

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Training array, containing the samples and their characteristics,
            Shape: (n_samples, n_features).
        y : Union[npt.NDArray, list]
            Array of target classes of ``X`` with (``n_samples``).
        verbose : bool
            Feedback on which sample aᵢ the memory cells are being generated.

        Returns
        -------
        AIRS
            Returns the instance itself.
        """
        X = self._prepare_features(X)
        y = check_array_type(y, "y")
        check_shape_match(X, y)

        self.classes = np.unique(y)
        sample_index = self._slice_index_list_by_class(y)
        progress = tqdm(
            total=len(y),
            postfix="\n",
            disable=not verbose,
            bar_format="{desc} ┇{bar}┇ {n}/{total} memory cells for each aᵢ",
        )
        pool_cells_classes = {}
        for _class_ in self.classes:
            progress.set_description_str(
                f"Generating the memory cells for the {_class_} class:"
            )

            x_class = X[sample_index[_class_]]

            self._cells_affinity_threshold(x_class)
            sufficiently_similar = self.affinity_threshold * self.affinity_threshold_scalar

            pool_c: list[BCell] = self._init_memory_c(x_class)

            for ai in x_class:
                c_match, match_stimulation = self._select_best_matching_cell(ai, pool_c)

                arb_list = self._generate_arb_list(ai, c_match, match_stimulation)
                c_candidate = self._refinement_arb(ai, match_stimulation, arb_list)

                if c_candidate.stimulation > match_stimulation:
                    pool_c.append(c_candidate.to_cell())
                    if (
                        self._affinity(c_candidate.vector, c_match.vector)
                        < sufficiently_similar
                    ):
                        pool_c.remove(c_match)

                progress.update()
            pool_cells_classes[_class_] = pool_c

        progress.set_description(
            f"\033[92m✔ Set of memory cells for classes ({', '.join(map(str, self.classes))}) "
            f"successfully generated\033[0m"
        )
        progress.close()
        self._cells_memory = pool_cells_classes
        self._all_class_cell_vectors = [
            (class_name, cell.vector)
            for class_name in self.classes
            for cell in self._cells_memory[class_name]
        ]
        return self

    def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
        """
        Predict class labels based on the memory cells created during training.

        This method uses the trained memory cells to perform classification of the input data
        using the k-nearest neighbors approach.

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Array with input samples with  Shape: (``n_samples, n_features``)

        Raises
        ------
        TypeError
            If X is not a ndarray or list.
        FeatureDimensionMismatch
            If the number of features in X does not match the expected number.
        ModelNotFittedError
            If the mode has not yet been adjusted and does not have defined memory cells, it is
            not able to predictions

        Returns
        -------
        C : npt.NDArray
            An ndarray of the form ``C`` (``n_samples``), containing the predicted classes for
            ``X``.
        """
        if self._all_class_cell_vectors is None:
            raise ModelNotFittedError("AIRS")

        X = check_array_type(X)
        check_feature_dimension(X, self._n_features)

        if self._feature_type == "binary-features":
            check_binary_array(X)

        return predict_knn_affinity(
            X, self.k, self._all_class_cell_vectors, self._affinity
        )

    def _select_best_matching_cell(
        self,
        ai: npt.NDArray,
        pool_c: list[BCell]
    ) -> tuple[BCell, float]:
        """Select the BCell with the highest affinity with antigen.

        Parameters
        ----------
        ai : npt.NDArray
            The current antigen.
        pool_c : list[BCell]
            Pool of memory B-Cells belonging to same class.

        Returns
        -------
        tuple[BCell, float]
            A tuple containing the best B cell and their affinity.
        """
        c_match = pool_c[0]
        match_stimulation = -1.0
        for cell in pool_c:
            stimulation = self._affinity(cell.vector, ai)
            if stimulation > match_stimulation:
                match_stimulation = stimulation
                c_match = cell

        return c_match, match_stimulation

    def _generate_arb_list(
        self,
        ai: npt.NDArray,
        c_match: BCell,
        match_stimulation: float
    ) -> list[_ARB]:
        """Generate a pool from the best affinity B cell.

        Parameters
        ----------
        ai : npt.NDArray
            The current antigen.
        c_match : BCell
            The best B-Cell
        match_stimulation : float
            The corresponding stimulation (affinity) value

        Returns
        -------
        list[_ARB]
            ARB set.
        """
        n_clones = int(self.rate_hypermutation * self.rate_clonal * match_stimulation)
        arb_list: list[_ARB] = [
            _ARB(vector=c_match.vector, stimulation=match_stimulation)
        ]

        if n_clones <= 0:
            return arb_list

        set_clones: npt.NDArray = c_match.hyper_clonal_mutate(
            n_clones,
            self._feature_type,
        )

        arb_list.extend(
            _ARB(vector=clone, stimulation=self._affinity(clone, ai))
            for clone in set_clones
        )

        return arb_list

    def _refinement_arb(
        self,
        ai: npt.NDArray,
        c_match_stimulation: float,
        arb_list: List[_ARB]
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
        .. [1] Brabazon, A., O'Neill, M., & McGarraghy, S. (2015).
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
                int(self.rate_clonal * c_match_stimulation), self._feature_type
            )

            arb_list = [
                _ARB(vector=clone, stimulation=self._affinity(clone, ai))
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

        > affinity_threshold = (Σᵢ=₁ⁿ⁻¹ Σⱼ=ᵢ₊₁ⁿ affinity(aᵢ, aⱼ)) / (n(n-1)/2)

        Parameters
        ----------
        antigens_list : npt.NDArray
            List of training antigens.
        """
        if self._feature_type == "binary-features":
            distances = pdist(antigens_list, metric="hamming")
        else:
            metric_kwargs = {"p": self.p} if self.metric == "minkowski" else {}
            distances = pdist(antigens_list, metric=self.metric, **metric_kwargs)  # type: ignore

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
        return float(1 - (distance / (1 + distance)))

    def _init_memory_c(self, antigens_list: npt.NDArray) -> List[BCell]:
        """
        Initialize memory cells by randomly selecting `rate_mc_init` antigens.

        Parameters
        ----------
        antigens_list : npt.NDArray
            List of training antigens.

        Returns
        -------
        List[BCell]
            List of initialized memories.
        """
        n = antigens_list.shape[0]
        n_cells = int(n * self.rate_mc_init)

        if n == 0 or n_cells == 0:
            return []

        permutation = np.random.permutation(n)
        selected = antigens_list[permutation[:n_cells]]
        return [BCell(ai) for ai in selected]

    def _prepare_features(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
        """
        Check the samples, specifying the type, quantity of characteristics, and other parameters.

        * This method updates the following attributes:
            * ``self._feature_type``
            * ``self.metric`` (only for binary features)
            * ``self._bounds`` (only for ranged features)
            * ``self._n_features``

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Training array, containing the samples and their characteristics,
            Shape: (n_samples, n_features).

        Returns
        -------
        X : npt.NDArray
            The processed input data.
        """
        X = check_array_type(X)
        self._feature_type = detect_vector_data_type(X)

        match self._feature_type:
            case "binary-features":
                X = X.astype(np.bool_)
                self.metric = "hamming"
            case "ranged-features":
                self._bounds = np.vstack(
                    [np.min(X, axis=0), np.max(X, axis=0)]
                )

        self._n_features = X.shape[1]

        return X
