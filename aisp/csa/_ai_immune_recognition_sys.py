"""Artificial Immune Recognition System (AIRS)"""

import random
from collections import Counter
from heapq import nlargest
from typing import List, Literal, Optional

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ._cs_core import binary_affinity_threshold, continuous_affinity_threshold
from ..utils.sanitizers import sanitize_param, sanitize_seed, sanitize_choice
from ..utils import slice_index_list_by_class
from ..utils.distance import hamming, compute_metric_distance, get_metric_code
from ..base.mutation import clone_and_mutate_continuous, clone_and_mutate_binary
from ._base import BaseAIRS


class _Cell:
    """Cell

    Represents a memory B-cell.

    Parameters:
    ----------
    * size (``Optional[int]``): The number of features in the vector. If `vector` is `None`, a
        random vector is generated. Defaults to None.
    * vector (``Optional[npt.NDArray]``): A vector of cell features. Defaults to None.
    * algorithm (``Literal["continuous-features", "binary-features"]``): The type of algorithm
        for continuous or binary samples. Defaults to "continuous-features".
    """

    def __init__(
        self,
        size: Optional[int] = None,
        vector: Optional[npt.NDArray] = None,
        algorithm: Literal[
            "continuous-features", "binary-features"
        ] = "continuous-features",
    ) -> None:
        if vector is None and size is not None:
            if algorithm == "binary-features":
                self.vector: npt.NDArray[np.bool_] = np.random.randint(0, 2, size=size).astype(
                    np.bool_
                )
            else:
                self.vector: npt.NDArray[np.float64] = np.random.uniform(size=size)
        else:
            self.vector: npt.NDArray = vector
        self.stimulation: float = 0
        self._algorithm: Literal["continuous-features", "binary-features"] = algorithm

    def hyper_clonal_mutate(self, n: int) -> npt.NDArray:
        """
        Clones N features from a cell's features, generating a set of mutated vectors.

        Parameters
        ----------
        * n (``int``): Number of clones to be generated from mutations of the original cell.

        Returns
        ----------
        * npt.NDArray: An array containing N mutated vectors from the original cell.
        """
        if self._algorithm == "binary-features":
            return clone_and_mutate_binary(self.vector, n)
        return clone_and_mutate_continuous(self.vector, n)


class _ABR(_Cell):
    """ABR (Artificial recognition ball)
    Individual from the set of recognizing cells (ABR), inherits characteristics from a B-cell,
    adding resource consumption

    Parameters
    ----------
    * size (``Optional[int]``): The number of features in the vector. If `vector` is `None`, a
        random vector is generated. Defaults to None.
    * vector (``Optional[npt.NDArray]``): A vector of cell features. Defaults to None.
    * stimulation (``Optional[float]``): The rate at which the cell stimulates antigens.
        Defaults to None.
    * algorithm (``Literal["continuous-features", "binary-features"]``): The type of algorithm
        for continuous or binary samples. Defaults to "continuous-features".
    """

    def __init__(
        self,
        size: Optional[int] = None,
        vector: Optional[npt.NDArray] = None,
        stimulation: Optional[float] = None,
        algorithm: Literal[
            "continuous-features", "binary-features"
        ] = "continuous-features",
    ) -> None:
        super().__init__(size=size, vector=vector, algorithm=algorithm)
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

        self._algorithm: Literal["continuous-features", "binary-features"] = (
            sanitize_param(
                algorithm, "continuous-features", lambda x: x == "binary-features"
            )
        )

        if algorithm == "binary-features":
            self.metric: str = "hamming"
        else:
            self.metric: str = sanitize_choice(metric, ["manhattan", "minkowski"], "euclidean")

        # Obtém as variáveis do kwargs.
        self.p: float = kwargs.get("p", 2)
        # Conjunto de células de memórias
        self.cells_memory = None
        self.affinity_threshold = 0.0
        self.classes = None

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

        super()._check_and_raise_exceptions_fit(X, y, self._algorithm)

        # Converte todo o array X para boolean quando utilizar a versão binária.
        if self._algorithm == "binary-features":
            X = X.astype(np.bool_)

        # Identificando as classes possíveis, dentro do array de saídas ``y``.
        self.classes = np.unique(y)
        # Separa as classes para o treinamento.
        sample_index = self.__slice_index_list_by_class(y)
        # Barra de progresso para o cada amostras (ai) em treinamento.
        if verbose:
            progress = tqdm(
                total=len(y),
                postfix="\n",
                bar_format="{desc} ┇{bar}┇ {n}/{total} memory cells for each aᵢ",
            )
        # Inicia o conjunto que receberá as células de memória.
        pool_cells_classes = {}
        for _class_ in self.classes:
            # Informando em qual classe o algoritmo está para a barra de progresso.
            if verbose:
                progress.set_description_str(
                    f"Generating the memory cells for the {_class_} class:"
                )
            x_class = X[sample_index[_class_]]
            # Calculando o limiar de semelhança entre os antígenos
            self._cells_affinity_threshold(x_class)
            # Iniciar as células de memória para uma classe.
            pool_c: list = self._init_memory_c(antigens_list=x_class)

            for ai in x_class:
                # Calculando o estimulo das células de memoria com aᵢ
                for cell in pool_c:
                    cell.stimulation = self._affinity(cell.vector, ai)

                # Pegando a célula com o maior estímulo do conjunto de memória e adicionando-a ao
                # conjunto ABR.
                c_match: _Cell = max(pool_c, key=lambda x: x.stimulation)
                abr_list: list[_ABR] = [
                    _ABR(
                        vector=c_match.vector,
                        stimulation=c_match.stimulation,
                        algorithm=self._algorithm,
                    )
                ]

                set_clones: npt.NDArray = c_match.hyper_clonal_mutate(
                    int(
                        self.rate_hypermutation * self.rate_clonal * c_match.stimulation
                    )
                )

                # Populando ARB com os clones
                for clone in set_clones:
                    abr_list.append(
                        _ABR(
                            vector=clone,
                            algorithm=self._algorithm,
                            stimulation=self._affinity(clone, ai),
                        )
                    )

                c_candidate = self._refinement_abr(ai, c_match, abr_list)

                if c_candidate.stimulation > c_match.stimulation:
                    pool_c.append(c_candidate)
                    if self._affinity(c_candidate.vector, c_match.vector) < (
                        self.affinity_threshold * self.affinity_threshold_scalar
                    ):
                        pool_c.remove(c_match)

                if verbose:
                    progress.update(1)
            pool_cells_classes[_class_] = pool_c
        # Informar a finalização das interações com os antígenos.
        if verbose:
            progress.set_description(
                f'\033[92m✔ Set of memory cells for classes ({", ".join(map(str, self.classes))}) '
                f"successfully generated\033[0m"
            )
        self.cells_memory = pool_cells_classes
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
            * C – (``npt.NDArray``): an ndarray of the form ``C`` [``N samples``],
            containing the predicted classes for ``X``.
            * ``None``: If there are no detectors for the prediction.

        ---

        Função para efetuar a previsão das classes com base nos detectores
        criados após o treinamento.

        Parameters:
        ---
            * X (``npt.NDArray``): Array com as amostras de entradas com [``N amostras`` (Linhas)] e
            [``N características``(Colunas)]

        Returns:
        ---
            * C – (``npt.NDArray``): um ndarray de forma ``C`` [``N amostras``],
            contendo as classes previstas para ``X``.
            * ``None``: Se não existir detectores para a previsão.
        """
        # se não houver detectores retorna None.
        if self.cells_memory is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, len(self.cells_memory[self.classes[0]][0].vector), self._algorithm
        )

        # Inicia uma lista vazia.
        c: list = []

        for line in X:
            label_stim_list = [
                (_class, self._affinity(cell.vector, line))
                for _class in self.classes
                for cell in self.cells_memory[_class]
            ]
            # Criar a lista com os k vizinhos mais próximos.
            k_nearest = nlargest(self.k, label_stim_list, key=lambda x: x[1])
            # Conta os votos com base no número de vezes que uma classe aparece na lista de knn.
            votes = Counter(label for label, _ in k_nearest)
            # Adiciona o rotulo com a maior quantidade de votos.
            c.append(votes.most_common(1)[0][0])
        return np.array(c)

    def _refinement_abr(
        self, ai: npt.NDArray, c_match: _Cell, abr_list: List[_ABR]
    ) -> _Cell:
        """
        Execute the refinement process for the ABR set until the average stimulation value exceeds
        the defined threshold (``affinity_threshold_scalar``).

        Parameters
        ----------
        * **c_match** (``_Cell``): Cell with the highest stimulation relative to aᵢ
        * **abr_list** (``List[_ABR]``): ABR set.

        Returns
        ----------
        * **_Cell**: The cell with the highest ABR stimulation
        """
        iters = 0
        # Competição e Refinamento ARB
        while True:
            iters += 1
            abr_list = sorted(abr_list, key=lambda x: x.stimulation, reverse=True)
            resource = self.n_resources
            for cell in abr_list:
                resource = cell.set_resource(
                    resource=resource, amplified=self.resource_amplified
                )
                if resource == 0:
                    break
            # remove as células sem recursos e calcula a media de estimulo de ABR.
            abr_list = list(filter(lambda item: item.resource != 0, abr_list))
            avg_stimulation = sum(item.stimulation for item in abr_list) / len(abr_list)
            # Se o máximo de interações ou a média do estímulo maior que o limiar para o loop
            if iters == self.max_iters or avg_stimulation > self.affinity_threshold:
                break

            # pegando uma célula aleatória e efetuando mutações.
            abr_random = random.choice(abr_list)
            clone_abr = abr_random.hyper_clonal_mutate(
                int(self.rate_clonal * c_match.stimulation)
            )

            # Adicionando os clones os ABR com a taxa de estimulo com aᵢ
            abr_list = [
                _ABR(
                    vector=clone,
                    stimulation=self._affinity(clone, ai),
                    algorithm=self._algorithm,
                )
                for clone in clone_abr
            ]

        # Retorna a célula com maior estímulo com aᵢ
        return max(abr_list, key=lambda x: x.stimulation)

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
        if self._algorithm == "binary-features":
            self.affinity_threshold = binary_affinity_threshold(antigens_list)
        else:
            self.affinity_threshold = continuous_affinity_threshold(
                antigens_list.astype(np.float64),  # Certifique-se do dtype!
                get_metric_code(self.metric),
                self.p
            )

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
        if self._algorithm == "binary-features":
            distance = hamming(u, v)
        else:
            distance = compute_metric_distance(u, v, get_metric_code(self.metric), self.p)
        return 1 - (distance / (1 + distance))

    def _init_memory_c(self, antigens_list: npt.NDArray) -> List[_Cell]:
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
        m_c = []
        n = round(len(antigens_list) * self.rate_mc_init)

        randomly_antigens_index = np.random.choice(
            antigens_list.shape[0], size=n, replace=False
        )
        for antigen in antigens_list[randomly_antigens_index]:
            m_c.append(_Cell(vector=antigen, algorithm=self._algorithm))
        return m_c

    def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
        """
        The function ``__slice_index_list_by_class(...)``, separates the indices of the lines \
        according to the output class, to loop through the sample array, only in positions where \
        the output is the class being trained.

        Parameters
        ----------
        * y (npt.NDArray): Receives a ``y``[``N sample``] array with the output classes of the \
            ``X`` sample array.

        returns
        ----------
        * dict: A dictionary with the list of array positions(``y``), with the classes as key.
        """
        return slice_index_list_by_class(self.classes, y)
