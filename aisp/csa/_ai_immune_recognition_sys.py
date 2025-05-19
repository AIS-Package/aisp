"""Artificial Immune Recognition System (AIRS)"""

import random
from collections import Counter
from heapq import nlargest
from operator import attrgetter
from typing import List, Literal, Optional, Dict

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from tqdm import tqdm

from ._cell import Cell
from ..utils.sanitizers import sanitize_param, sanitize_seed, sanitize_choice
from ..utils.distance import hamming, compute_metric_distance, get_metric_code
from ._base import BaseAIRS


class _ABR(Cell):
    """ABR (Artificial recognition ball)
    Individual from the set of recognizing cells (ABR), inherits characteristics from a B-cell,
    adding resource consumption

    Parameters
    ----------
    * vector (``Optional[npt.NDArray]``): A vector of cell features. Defaults to None.
    * stimulation (``Optional[float]``): The rate at which the cell stimulates antigens.
        Defaults to None.
    """

    def __init__(
        self,
        vector: Optional[npt.NDArray] = None,
        stimulation: Optional[float] = None
    ) -> None:
        super().__init__(vector)
        self.resource = 0
        if stimulation is not None:
            self.stimulation: float = stimulation

    def set_resource(self, resource: float, amplified: float = 1) -> float:
        """
        Updates the amount of resources available for an ABR after consumption.

        This function consumes the resources and returns the remaining amount of resources after
        consumption.

        Parameters
        ----------
        * resource (``float``): Initial amount of resources.
        * amplified (``float``): Amplifier for the resource consumption by the cell. It is
            multiplied by the cell's stimulus. The default value is 1.

        Returns
        ----------
        * (``float``): The remaining amount of resources after consumption.
        """
        aux_resource = resource - (self.stimulation * amplified)
        if aux_resource < 0:
            self.resource = resource
            return 0

        self.resource = aux_resource
        return aux_resource

    def to_cell(self) -> Cell:
        """Converte este _ABR em um objeto Cell puro."""
        return Cell(self.vector)


class AIRS(BaseAIRS):
    """Artificial Immune Recognition System (AIRS)

    The AIRS is a classification algorithm inspired by the clonal selection process. The \
    version implemented in this class is inspired by its simplified version, AIRS2, described in \
    [Brabazon, O'Neill, and McGarraghy (2015)](https://doi.org/10.1007/978-3-662-43631-8). \
    In this class, there is an adaptation for real-valued data and a secondary option for binary \
    features.

    Parameters
    ----------
    * n_resources (``float``): Total amount of available resources. Defaults to 10.
    * rate_clonal (``float``): Maximum number of possible clones of a class. This \
        quantity is multiplied by (cell stimulus * rate_hypermutation) to define the number
        of clones. Defaults to 10.
    * rate_mc_init (``float``): Percentage of samples used to initialize memory cells.
    * rate_hypermutation (``float``): The rate of mutated clones derived from rate_clonal as a
        scalar factor. Defaults to 0.75.
    * affinity_threshold_scalar (``float``): Normalized affinity threshold. Defaults to 0.75.
    * k (``int``): The number of K nearest neighbors that will be used to choose a label \
        in the prediction. Defaults to 10.
    * max_iters (``int``): Maximum number of interactions in the refinement process of \
        the ABR set exposed to aᵢ. Defaults to 100.
    * resource_amplified (``float``): Resource consumption amplifier is multiplied with \
        the incentive to subtract resources. Defaults to 1.0 without amplification.
    * metric (Literal["manhattan", "minkowski", "euclidean"]): Way to calculate the \
        distance between the detector and the sample: \

        * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression: \
        √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
        * ``'minkowski'`` ➜ The calculation of the distance is given by the expression: \
        ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
        * ``'manhattan'`` ➜ The calculation of the distance is given by the expression: \
        ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|). \
        Defaults to "Euclidean".

    * algorithm (Literal["continuous-features", "binary-features"]): [description]. \
        Defaults to "continuous-features".
    * seed (int): Seed for the random generation of detector values. Defaults to None.

    - ``**kwargs``:
        - p (``float``): This parameter stores the value of ``p`` used in the Minkowski \
        distance. The default is ``2``, which represents normalized Euclidean distance.\
        Different values of p lead to different variants of the [Minkowski Distance][1].

    Notes
    ----------
    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.minkowski_distance.html

    """

    def __init__(
        self,
        n_resources: float = 10,
        rate_clonal: int = 10,
        rate_mc_init: float = 0.2,
        rate_hypermutation: float = 0.75,
        affinity_threshold_scalar: float = 0.75,
        k: int = 10,
        max_iters: int = 100,
        resource_amplified: float = 1.0,
        metric: Literal["manhattan", "minkowski", "euclidean"] = "euclidean",
        algorithm: Literal[
            "continuous-features", "binary-features"
        ] = "continuous-features",
        seed: int = None,
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

        # Obtém as variáveis do kwargs.
        self.p: np.float64 = np.float64(kwargs.get("p", 2.0))
        # Conjunto de células de memórias
        self._cells_memory = None
        self.affinity_threshold = 0.0
        self.classes = None

    @property
    def cells_memory(self) -> Dict[str, list[Cell]]:
        """Returns the trained cells memory, organized by class."""
        return self._cells_memory

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        The function ``fit(...)``, performs the training according to ``X`` and ``y``, using the
        method Artificial Immune Recognition System (``AIRS``).

        Parameters
        ----------
        * X (``npt.NDArray``): Training array, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].
        * y (``npt.NDArray``): Array of target classes of ``X`` with [``N samples`` (lines)].
        * verbose (``bool``): Feedback on which sample aᵢ the memory cells are being generated.

        returns
        ----------
        * (``self``): Returns the instance itself.
        """
        progress = None

        super()._check_and_raise_exceptions_fit(X, y, self.algorithm)

        if self.algorithm == "binary-features":
            X = X.astype(np.bool_)

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
            if verbose:
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
                c_match = None
                match_stimulation = -1
                for cell in pool_c:
                    stimulation = self._affinity(cell.vector, ai)
                    if stimulation > match_stimulation:
                        match_stimulation = stimulation
                        c_match = cell

                abr_list: list[_ABR] = [
                    _ABR(
                        vector=c_match.vector,
                        stimulation=match_stimulation
                    )
                ]

                set_clones: npt.NDArray = c_match.hyper_clonal_mutate(
                    int(self.rate_hypermutation * self.rate_clonal * match_stimulation),
                    self.algorithm
                )

                for clone in set_clones:
                    abr_list.append(
                        _ABR(
                            vector=clone,
                            stimulation=self._affinity(clone, ai),
                        )
                    )

                c_candidate = self._refinement_abr(ai, match_stimulation, abr_list)

                if c_candidate.stimulation > match_stimulation:
                    pool_c.append(c_candidate.to_cell())
                    if (
                        self._affinity(c_candidate.vector, c_match.vector)
                        < sufficiently_similar
                    ):
                        pool_c.remove(c_match)

                if verbose:
                    progress.update(1)
            pool_cells_classes[_class_] = pool_c

        if verbose:
            progress.set_description(
                f"\033[92m✔ Set of memory cells for classes ({', '.join(map(str, self.classes))}) "
                f"successfully generated\033[0m"
            )
        self._cells_memory = pool_cells_classes
        return self

    def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]:
        """
        Function to perform the prediction of classes based on detectors
        created after training.

        Parameters:
        ---
        * X (``npt.NDArray``): Array with input samples with [``N samples`` (Lines)] and
            [``N characteristics``(Columns)]

        returns:
        ---
        * C – (``npt.NDArray``): an ndarray of the form ``C`` [``N samples``], containing the
            predicted classes for ``X``.
        * ``None``: If there are no detectors for the prediction.
        """
        if self._cells_memory is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, len(self._cells_memory[self.classes[0]][0].vector), self.algorithm
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

    def _refinement_abr(
        self, ai: npt.NDArray, c_match_stimulation: float, abr_list: List[_ABR]
    ) -> _ABR:
        """
        Execute the refinement process for the ABR set until the average stimulation value exceeds
        the defined threshold (``affinity_threshold_scalar``).

        Parameters
        ----------
        * c_match (``_Cell``): Cell with the highest stimulation relative to aᵢ
        * abr_list (``List[_ABR]``): ABR set.

        Returns
        ----------
        * _ABR: The cell with the highest ABR stimulation
        """
        iters = 0
        while True:
            iters += 1
            abr_list.sort(key=attrgetter("stimulation"), reverse=True)
            resource = self.n_resources
            for abr in abr_list:
                resource = abr.set_resource(
                    resource=resource, amplified=self.resource_amplified
                )
                if resource == 0:
                    break
            # remove cells without resources and calculate the average ABR stimulus.
            abr_list = [cell for cell in abr_list if cell.resource > 0]
            if not abr_list:
                break
            avg_stimulation = sum(item.stimulation for item in abr_list) / len(abr_list)

            if iters == self.max_iters or avg_stimulation > self.affinity_threshold:
                break

            # pick a random cell for mutations.
            random_index = random.randint(0, len(abr_list) - 1)
            clone_abr = abr_list[random_index].hyper_clonal_mutate(
                int(self.rate_clonal * c_match_stimulation),
                self.algorithm
            )

            abr_list = [
                _ABR(
                    vector=clone,
                    stimulation=self._affinity(clone, ai)
                )
                for clone in clone_abr
            ]

        return max(abr_list, key=attrgetter("stimulation"))

    def _cells_affinity_threshold(self, antigens_list: npt.NDArray):
        """
        This function calculates the affinity threshold based on the average affinity between
        training instances, where aᵢ and aⱼ are a pair of antigens, and affinity
        is measured by distance (Euclidean, Manhattan, Minkowski, Hamming).
        Following the formula:

        > affinity_threshold = (Σᵢ=₁ⁿ⁻¹ Σⱼ=ᵢ₊₁ⁿ affinity(aᵢ, aⱼ)) / (n(n-1)/2

        Parameters
        ----------
        - antigens_list (``NDArray``): List of training antigens.
        """
        if self.algorithm == "binary-features":
            distances = pdist(antigens_list, metric="hamming")
        elif self.metric == "minkowski":
            distances = pdist(antigens_list, metric="minkowski", p=self.p)
        else:
            distances = pdist(antigens_list, metric=self.metric)
        n = antigens_list.shape[0]
        sum_affinity = np.sum(1.0 - (distances / (1.0 + distances)))
        self.affinity_threshold = 1.0 - (sum_affinity / ((n * (n - 1)) / 2))

    def _affinity(self, u: npt.NDArray, v: npt.NDArray) -> float:
        """
        Calculates the stimulus between two vectors using metrics.

        Parameters
        ----------
        * u (``npt.NDArray``): Coordinates of the first point.
        * v (``npt.NDArray``): Coordinates of the second point.

        returns
        ----------
        * (``float``) the stimulus rate between the vectors.
        """
        distance: float
        if self.algorithm == "binary-features":
            distance = hamming(u, v)
        else:
            distance = compute_metric_distance(
                u, v, get_metric_code(self.metric), self.p
            )
        return 1 - (distance / (1 + distance))

    def _init_memory_c(self, antigens_list: npt.NDArray) -> List[Cell]:
        """
        This function initializes memory cells by randomly selecting `rate_mc_init`
        from the list of training antigens.

        Parameters
        ----------
        - antigens_list (``NDArray``): List of training antigens.

        Returns
        ----------
        * Mc: List of initialized memories.
        """
        n = antigens_list.shape[0]
        n_cells = int(n * self.rate_mc_init)

        permutation = np.random.permutation(n)
        selected = antigens_list[permutation[:n_cells]]
        return [Cell(ai) for ai in selected]
