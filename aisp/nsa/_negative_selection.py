"""Negative Selection Algorithm."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union, List

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ._base import check_detector_rnsa_validity
from ..base import BaseClassifier
from ..base.immune.cell import Detector
from ..exceptions import MaxDiscardsReachedError, ModelNotFittedError
from ..utils.distance import (
    min_distance_to_class_vectors,
    get_metric_code,
    compute_metric_distance,
)
from ..utils.random import set_seed_numba
from ..utils.sanitizers import sanitize_seed, sanitize_choice, sanitize_param
from ..utils.validation import (
    check_array_type,
    check_shape_match,
    check_feature_dimension,
)


class RNSA(BaseClassifier):
    """Real-Valued Negative Selection Algorithm (RNSA).

    Algorithm for classification and anomaly detection Based on self or not self
    discrimination, inspired by Negative Selection Algorithm.

    Parameters
    ----------
    N : int, default=100
        Number of detectors.
    r : float, default=0.05
        Radius of the detector.
    r_s : float, default=0.0001
        rₛ Radius of the ``X`` own samples.
    k : int, default=1
        Number of neighbors near the randomly generated detectors to perform the distance average
        calculation.
    metric: {"euclidean", "minkowski", "manhattan"}, default='euclidean'
        Distance metric used to compute the distance between the detector and the sample.
    max_discards : int, default=1000
        This parameter indicates the maximum number of consecutive detector discards, aimed at
        preventing a possible infinite loop in case a radius is defined that cannot generate
        non-self detectors.
    seed : int, default=None
        Seed for the random generation of values in the detectors.
    algorithm : {"default-NSA", "V-detector"}, default='default-NSA'
        Set the algorithm version:  
        * ``'default-NSA'``: Default algorithm with fixed radius.
        * ``'V-detector'``: This algorithm is based on the article Ji & Dasgupta (2004) [1]_
            and uses a variable radius for anomaly detection in feature spaces.

    **kwargs : dict
        Additional parameters. The following arguments are recognized:  
        * non_self_label : str, default='non-self'
            This variable stores the label that will be assigned when the data has only one
            output class, and the sample is classified as not belonging to that class.
        * cell_bounds : bool, default=False
            If set to ``True``, this option limits the generation of detectors to the space
            within the plane between 0 and 1. This means that any detector whose radius exceeds
            this limit is discarded, this variable is only used in the ``V-detector`` algorithm.
        * p : float, default=2
            This parameter stores the value of ``p`` used in the Minkowski distance. The default
            is ``2``, which represents Euclidean distance. Different values of p lead
            to different variants of the Minkowski Distance.

    Attributes
    ----------
    detectors : Optional[Dict[str | int, list[Detector]]]
        The trained detectors, organized by class.

    Warnings
    --------
    The parameters `r` and `r_s` can prevent the generation of valid detectors. A very small `r`
    value can limit coverage, while a very high one can hinder the generation of valid detectors.
    Similarly, a high r_s can restrict detector creation. Thus, proper adjustment of `r` and `r_s`
    is essential to ensure good model performance.

    Notes
    -----
    This algorithm has two different versions: one based on the canonical version [1] and another
    with variable radius detectors [2]. Both are adapted to work with multiple classes and have
    methods for predicting data present in the non-self region of all detectors and classes.

    References
    ----------
    .. [1] BRABAZON, Anthony; O'NEILL, Michael; MCGARRAGHY, Seán. Natural Computing
        Algorithms. [S. l.]: Springer Berlin Heidelberg, 2015. DOI 10.1007/978-3-662-43631-8.
        Disponível em: https://dx.doi.org/10.1007/978-3-662-43631-8.
    .. [2] Ji, Z.; Dasgupta, D. (2004).
        Real-Valued Negative Selection Algorithm with Variable-Sized Detectors.
        In *Lecture Notes in Computer Science*, vol. 3025.
        https://doi.org/10.1007/978-3-540-24854-5_30

    Examples
    --------
    >>> import numpy as np
    >>> from aisp.nsa import RNSA

    >>> np.random.seed(1)
    >>> class_a = np.random.uniform(high=0.5, size=(50, 2))
    >>> class_b = np.random.uniform(low=0.51, size=(50, 2))

    Example 1: Multiclass classification (RNSA supports two or more classes)
    
    >>> x_train = np.vstack((class_a, class_b))
    >>> y_train = ['a'] * 50 + ['b'] * 50
    >>> rnsa = RNSA(N=150, r=0.3, seed=1)
    >>> rnsa = rnsa.fit(x_train, y_train, verbose=False)
    >>> x_test = [
    ...     [0.15, 0.45],  # Expected: Class 'a'
    ...     [0.85, 0.65],  # Esperado: Classe 'b'
    ... ]
    >>> y_pred = rnsa.predict(x_test)
    >>> print(y_pred)
    ['a' 'b']

    Example 2: Anomaly Detection (self/non-self)
    
    >>> rnsa = RNSA(N=150, r=0.3, seed=1)
    >>> rnsa = rnsa.fit(X=class_a, y=np.array(['self'] * 50), verbose=False)
    >>> y_pred = rnsa.predict(class_b[:5])
    >>> print(y_pred)
    ['non-self' 'non-self' 'non-self' 'non-self' 'non-self']
    """

    def __init__(
        self,
        N: int = 100,
        r: float = 0.05,
        r_s: float = 0.0001,
        k: int = 1,
        metric: Literal["manhattan", "minkowski", "euclidean"] = "euclidean",
        max_discards: int = 1000,
        seed: Optional[int] = None,
        algorithm: Literal["default-NSA", "V-detector"] = "default-NSA",
        **kwargs: Any,
    ):
        self.metric: str = sanitize_choice(
            metric, ["manhattan", "minkowski"], "euclidean"
        )
        self.seed: Optional[int] = sanitize_seed(seed)
        if self.seed is not None:
            np.random.seed(seed)
            set_seed_numba(self.seed)
        self.k: int = sanitize_param(k, 1, lambda x: x > 1)
        self.N: int = sanitize_param(N, 100, lambda x: x >= 1)
        self.r: float = sanitize_param(r, 0.05, lambda x: x > 0)
        self.r_s: float = sanitize_param(r_s, 0.0001, lambda x: x > 0)
        self.algorithm: str = sanitize_param(
            algorithm, "default-NSA", lambda x: x == "V-detector"
        )
        self.max_discards: int = sanitize_param(max_discards, 1000, lambda x: x > 0)

        # Retrieves the variables from kwargs.
        self.p: np.float64 = np.float64(kwargs.get("p", 2))
        self.cell_bounds: bool = bool(kwargs.get("cell_bounds", False))
        self.non_self_label: str = str(kwargs.get("non_self_label", "non-self"))

        # Initializes the other class variables as None.
        self._detectors: Optional[Dict[str | int, list[Detector]]] = None
        self.classes: Optional[npt.NDArray] = None

    @property
    def detectors(self) -> Optional[Dict[str | int, list[Detector]]]:
        """Returns the trained detectors, organized by class."""
        return self._detectors

    def fit(
        self,
        X: Union[npt.NDArray, list],
        y: Union[npt.NDArray, list],
        verbose: bool = True,
    ) -> RNSA:
        """
        Perform training according to X and y, using the negative selection method (NegativeSelect).

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Training array, containing the samples and their characteristics.
            Shape: ``(n_samples, n_features)``
        y : Union[npt.NDArray, list]
            Array of target classes of ``X`` with ``n_samples`` (lines).
        verbose: bool, default=True
            Feedback from detector generation to the user.

        Raises
        ------
        TypeError
            If X or y are not ndarrays or have incompatible shapes.
        MaxDiscardsReachedError
            The maximum number of detector discards was reached during maturation. Check the
            defined radius value and consider reducing it.

        Returns
        -------
        self : RNSA
        Returns the instance itself.
        """
        X = check_array_type(X)
        y = check_array_type(y, "y")
        check_shape_match(X, y)
        self._n_features = X.shape[1]

        # Identifying the possible classes within the output array `y`.
        self.classes = np.unique(y)
        # Dictionary that will store detectors with classes as keys.
        list_detectors_by_class = {}
        # Separates the classes for training.
        sample_index = self._slice_index_list_by_class(y)
        # Progress bar for generating all detectors.
        progress = tqdm(
            total=int(self.N * (len(self.classes))),
            bar_format="{desc} ┇{bar}┇ {n}/{total} detectors",
            postfix="\n",
            disable=not verbose,
        )
        for _class_ in self.classes:
            # Initializes the empty set that will contain the valid detectors.
            valid_detectors_set: List[Detector] = []
            discard_count = 0
            x_class = X[sample_index[_class_]]
            # Indicating which class the algorithm is currently processing for the progress bar.
            progress.set_description_str(
                f"Generating the detectors for the {_class_} class:"
            )
            while len(valid_detectors_set) < self.N:
                # Generates a candidate detector vector randomly with values between 0 and 1.
                vector_x = np.random.random_sample(size=(self._n_features,))
                # Checks the validity of the detector for non-self with respect to the class samples
                valid_detector = self._checks_valid_detector(x_class, vector_x)

                # If the detector is valid, add it to the list of valid detectors.
                if valid_detector is not False:
                    discard_count = 0
                    radius: Optional[float] = None
                    if self.algorithm == "V-detector" and isinstance(
                        valid_detector, tuple
                    ):
                        radius = valid_detector[1]
                    valid_detectors_set.append(Detector(vector_x, radius))
                    progress.update()
                else:
                    discard_count += 1
                    if discard_count == self.max_discards:
                        raise MaxDiscardsReachedError(_class_)

            # Add detectors, with classes as keys in the dictionary.
            list_detectors_by_class[_class_] = valid_detectors_set
        # Notify completion of detector generation for the classes.
        progress.set_description(
            f"\033[92m✔ Non-self detectors for classes ({', '.join(map(str, self.classes))}) "
            f"successfully generated\033[0m"
        )
        progress.close()
        # Saves the found detectors in the attribute for the non-self detectors of the trained model
        self._detectors = list_detectors_by_class
        return self

    def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
        """
        Prediction of classes based on detectors created after training.

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Array with input samples with Shape: (n_samples, n_features)

        Raises
        ------
        TypeError
            If X is not a ndarray or list.
        FeatureDimensionMismatch
            If the number of features in X does not match the expected number.
        ModelNotFittedError
            If the mode has not yet been adjusted and does not have defined detectors or
            classes, it is not able to predictions

        Returns
        -------
        C : npt.NDArray
            A ndarray of the form ``C`` (n_samples), containing the predicted classes
            for ``X``.
        """
        if self._detectors is None or self.classes is None:
            raise ModelNotFittedError("RNSA")
        X = check_array_type(X)
        check_feature_dimension(X, self._n_features)

        # Initializes an empty array that will store the predictions.
        c = []
        # For each sample row in X.
        for line in X:
            class_found: bool
            _class_ = self._compare_sample_to_detectors(line)
            if _class_ is None:
                class_found = False
            else:
                c.append(_class_)
                class_found = True

            # If there is only one class and the sample is not classified,
            # set the output as non-self.
            if not class_found and len(self.classes) == 1:
                c.append(self.non_self_label)
            # If the class is not identified with the detectors, assign the class with
            # the greatest distance from the mean of its detectors.
            elif not class_found:
                average_distance: dict = {}
                for _class_ in self.classes:
                    detectores = [x.position for x in self._detectors[_class_]]
                    average_distance[_class_] = np.average(
                        [self._distance(detector, line) for detector in detectores]
                    )
                c.append(max(average_distance, key=average_distance.get))  # type: ignore
        return np.array(c)

    def _checks_valid_detector(
        self,
        x_class: npt.NDArray,
        vector_x: npt.NDArray
    ) -> Union[bool, tuple[bool, float]]:
        """
        Check if the detector has a valid non-proper r radius for the class.

        Parameters
        ----------
        x_class : npt.NDArray
            Array ``x_class`` with the samples per class.
        vector_x : npt.NDArray
            Randomly generated vector x candidate detector with values between[0, 1].

        Returns
        -------
        is_valid : Union[bool, tuple[bool, float]]
            Returns whether the detector is valid or not.
        """
        # If any of the input arrays have zero size, Returns false.
        if np.size(x_class) == 0 or np.size(vector_x) == 0:
            return False
        # If self.k > 1, uses the k nearest neighbors (kNN); otherwise, checks the detector
        # without considering kNN.
        if self.k > 1:
            knn_list: list = []
            for x in x_class:
                # Calculates the distance between the two vectors and adds it to the kNN list if
                # the distance is smaller than the largest distance in the list.
                self._compare_knearest_neighbors_list(
                    knn_list, self._distance(x, vector_x)
                )
            # If the average of the distances in the kNN list is less than the radius, Returns true.
            distance_mean = np.mean(knn_list)
            if self.algorithm == "V-detector":
                return self._detector_is_valid_to_vdetector(
                    float(distance_mean), vector_x
                )
            if distance_mean > (self.r + self.r_s):
                return True
        else:
            if self.algorithm == "V-detector":
                distance = min_distance_to_class_vectors(
                    x_class, vector_x, get_metric_code(self.metric), self.p
                )
                return self._detector_is_valid_to_vdetector(distance, vector_x)

            # Calculates the distance between the vectors; if not it is less than or equal to
            # the radius plus the sample's radius, sets the validity of the detector to
            # true.
            threshold: float = self.r + self.r_s
            if check_detector_rnsa_validity(
                x_class, vector_x, threshold, get_metric_code(self.metric), self.p
            ):
                return True  # Detector is valid!

        return False  # Detector is not valid!

    def _compare_knearest_neighbors_list(self, knn: list, distance: float) -> None:
        """
        Compare the k-nearest neighbor distance at position k=1 in the list knn.

        If the distance of the new sample is less, replace it and sort in ascending order.

        Parameters
        ----------
        knn : list
            List of k-nearest neighbor distances.
        distance : float
            Distance to check.
        """
        # If the number of distances in kNN is less than k, adds the distance.
        if len(knn) < self.k:
            knn.append(distance)
            knn.sort()
        # Otherwise, add the distance if the new distance is smaller than the largest
        # distance in the list.
        elif knn[self.k - 1] > distance:
            knn[self.k - 1] = distance
            knn.sort()

    def _compare_sample_to_detectors(self, line: npt.NDArray) -> Optional[str]:
        """
        Compare a sample with the detectors, verifying if the sample is proper.

        Parameters
        ----------
        line : npt.NDArray
            vector with N-features

        Returns
        -------
        possible_classes : Optional[str]
            Returns the predicted class with the detectors or None if the sample does not qualify
            for any class.
        """
        if self._detectors is None or self.classes is None:
            return None

        # List to store the classes and the average distance between the detectors and the sample.
        possible_classes = []
        for _class_ in self.classes:
            # Variable to indicate if the class was found with the detectors.
            class_found: bool = True
            sum_distance = 0.0
            for detector in self._detectors[_class_]:
                distance = self._distance(detector.position, line)
                sum_distance += distance
                if self.algorithm == "V-detector" and detector.radius is not None:
                    if distance <= detector.radius:
                        class_found = False
                        break
                elif distance <= self.r:
                    class_found = False
                    break

            # If the sample passes through all the detectors of a class, adds the class as a
            # possible prediction.
            if class_found:
                possible_classes.append([_class_, sum_distance / self.N])
        # If classified as belonging to only one class, Returns the class.
        if len(possible_classes) == 1:
            return possible_classes[0][0]
        # If belonging to more than one class, Returns the class with the greatest average distance.
        if len(possible_classes) > 1:
            return max(possible_classes, key=lambda x: x[1])[0]

        return None

    def _distance(self, u: npt.NDArray, v: npt.NDArray) -> float:
        """
        Calculate the distance between two points by the chosen ``metric``.

        Parameters
        ----------
        u : npt.NDArray
            Coordinates of the first point.
        v : npt.NDArray
            Coordinates of the second point.

        Returns
        -------
        Distance : float
            between the two points.
        """
        return compute_metric_distance(u, v, get_metric_code(self.metric), self.p)

    def _detector_is_valid_to_vdetector(
        self,
        distance: float,
        vector_x: npt.NDArray
    ) -> Union[bool, tuple[bool, float]]:
        """Validate the detector against the vdetector.

        Check if the distance between the detector and the samples, minus the radius of the
        samples, is greater than the minimum radius.

        Parameters
        ----------
        distance : float
            minimum distance calculated between all samples.
        vector_x : np.ndarray
            randomly generated candidate detector vector x with values between 0 and 1.

        Returns
        -------
        valid : bool

            - ``False`` if the calculated radius is smaller than the minimum distance or exceeds
            the edge of the space, if this option is enabled.

            - ``True`` and the distance minus the radius of the samples, if the radius is valid.`
        """
        new_detector_r = float(distance - self.r_s)
        if self.r >= new_detector_r:
            return False

        # If _cell_bounds is True, considers the detector to be within the plane bounds.
        if self.cell_bounds:
            for p in vector_x:
                if (p - new_detector_r) < 0 or (p + new_detector_r) > 1:
                    return False

        return True, new_detector_r
