"""Negative Selection Algorithm."""

from typing import Dict, Literal, Optional, Union
from tqdm import tqdm

import numpy as np
import numpy.typing as npt

from ._ns_core import (
    check_detector_bnsa_validity,
    bnsa_class_prediction,
    check_detector_rnsa_validity,
)
from ..exceptions import MaxDiscardsReachedError
from ..utils.distance import (
    min_distance_to_class_vectors,
    get_metric_code,
    compute_metric_distance,
)
from ..utils.sanitizers import sanitize_seed, sanitize_choice, sanitize_param
from ._base import BaseNSA, Detector


class RNSA(BaseNSA):
    """Real-Valued Negative Selection Algorithm (RNSA) for classification and anomaly detection.

    Uses the self and non-self method to identify anomalies.

    Attributes
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
    metric: str, default='euclidean'
        Way to calculate the distance between the detector and the sample:

        + ``'Euclidean'`` ➜ The calculation of the distance is given by the expression:
            √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
        + ``'minkowski'`` ➜ The calculation of the distance is given by the expression:
            ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
        + ``'manhattan'`` ➜ The calculation of the distance is given by the expression:
            ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) .
    max_discards : int, default=1000
        This parameter indicates the maximum number of consecutive detector discards, aimed at
        preventing a possible infinite loop in case a radius is defined that cannot generate
        non-self detectors.
    seed : int, default=None
        Seed for the random generation of values in the detectors.
    algorithm : str, default='default-NSA'
        Set the algorithm version:

        * ``'default-NSA'``: Default algorithm with fixed radius.
        * ``'V-detector'``: This algorithm is based on the article
            [Real-Valued Negative Selection Algorithm with Variable-Sized Detectors][2], by Ji,
            Z., Dasgupta, D. (2004), and uses a variable radius for anomaly detection in feature
            spaces.
    **kwargs : dict
        Parâmetros adicionais. Os seguintes argumentos são reconhecidos:

        - non_self_label : str, default='non-self'
            This variable stores the label that will be assigned when the data has only one
            output class, and the sample is classified as not belonging to that class.
        - cell_bounds : bool, default=False
            If set to ``True``, this option limits the generation of detectors to the space
            within the plane between 0 and 1. This means that any detector whose radius exceeds
            this limit is discarded, this variable is only used in the ``V-detector`` algorithm.
        - p : float, default=2
            This parameter stores the value of ``p`` used in the Minkowski distance. The default
            is ``2``, which represents normalized Euclidean distance. Different values of p lead
            to different variants of the [Minkowski Distance][1].

    Notes
    -----
    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.minkowski_distance.html

    [2] https://doi.org/10.1007/978-3-540-24854-5_30

    """

    def __init__(
        self,
        N: int = 100,
        r: float = 0.05,
        r_s: float = 0.0001,
        k: int = 1,
        metric: Literal["manhattan", "minkowski", "euclidean"] = "euclidean",
        max_discards: int = 1000,
        seed: int = None,
        algorithm: Literal["default-NSA", "V-detector"] = "default-NSA",
        **kwargs: Dict[str, Union[bool, str, float]],
    ):
        self.metric = sanitize_choice(metric, ["manhattan", "minkowski"], "euclidean")
        self.seed = sanitize_seed(seed)
        if self.seed is not None:
            np.random.seed(seed)
        self.k: int = sanitize_param(k, 1, lambda x: x > 1)
        self.N: int = sanitize_param(N, 100, lambda x: x >= 1)
        self.r: float = sanitize_param(r, 0.05, lambda x: x > 0)
        self.r_s: float = sanitize_param(r_s, 0.0001, lambda x: x > 0)
        self.algorithm: str = sanitize_param(
            algorithm, "default-NSA", lambda x: x == "V-detector"
        )
        self.max_discards: int = sanitize_param(max_discards, 1000, lambda x: x > 0)

        # Retrieves the variables from kwargs.
        self.p: float = kwargs.get("p", 2)
        self.cell_bounds: bool = kwargs.get("cell_bounds", False)
        self.non_self_label: str = kwargs.get("non_self_label", "non-self")

        # Initializes the other class variables as None.
        self._detectors: Union[dict, None] = None
        self.classes: npt.NDArray = None

    @property
    def detectors(self) -> Dict[str, list[Detector]]:
        """Returns the trained detectors, organized by class."""
        return self._detectors

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        Perform training according to X and y, using the negative selection method (NegativeSelect).

        Parameters
        ----------
        X : npt.NDArray
            Training array, containing the samples and their characteristics, [``N samples`` (
            rows)][``N features`` (columns)].
        y : npt.NDArray
            Array of target classes of ``X`` with [``N samples`` (lines)].
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
        progress = None
        super()._check_and_raise_exceptions_fit(X, y)

        # Identifying the possible classes within the output array `y`.
        self.classes = np.unique(y)
        # Dictionary that will store detectors with classes as keys.
        list_detectors_by_class = {}
        # Separates the classes for training.
        sample_index = self._slice_index_list_by_class(y)
        # Progress bar for generating all detectors.
        if verbose:
            progress = tqdm(
                total=int(self.N * (len(self.classes))),
                bar_format="{desc} ┇{bar}┇ {n}/{total} detectors",
                postfix="\n",
            )
        for _class_ in self.classes:
            # Initializes the empty set that will contain the valid detectors.
            valid_detectors_set = []
            discard_count = 0
            x_class = X[sample_index[_class_]]
            # Indicating which class the algorithm is currently processing for the progress bar.
            if verbose:
                progress.set_description_str(
                    f"Generating the detectors for the {_class_} class:"
                )
            while len(valid_detectors_set) < self.N:
                # Generates a candidate detector vector randomly with values between 0 and 1.
                vector_x = np.random.random_sample(size=X.shape[1])
                # Checks the validity of the detector for non-self with respect to the class samples
                valid_detector = self.__checks_valid_detector(x_class, vector_x)

                # If the detector is valid, add it to the list of valid detectors.
                if valid_detector is not False:
                    discard_count = 0
                    radius = (
                        valid_detector[1] if self.algorithm == "V-detector" else None
                    )
                    valid_detectors_set.append(Detector(vector_x, radius))
                    if verbose:
                        progress.update(1)
                else:
                    discard_count += 1
                    if discard_count == self.max_discards:
                        raise MaxDiscardsReachedError(_class_)

            # Add detectors, with classes as keys in the dictionary.
            list_detectors_by_class[_class_] = valid_detectors_set
        # Notify completion of detector generation for the classes.
        if verbose:
            progress.set_description(
                f"\033[92m✔ Non-self detectors for classes ({', '.join(map(str, self.classes))}) "
                f"successfully generated\033[0m"
            )
        # Saves the found detectors in the attribute for the non-self detectors of the trained model
        self._detectors = list_detectors_by_class
        return self

    def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]:
        """
        Prediction of classes based on detectors created after training.

        Parameters
        ----------
        X : npt.NDArray
            Array with input samples with [``N_samples`` (Lines)] and [``N_characteristics``
            (Columns)]

        Raises
        ------
        TypeError
            If X is not an ndarray or list.
        FeatureDimensionMismatch
            If the number of features in X does not match the expected number.

        Returns
        -------
        C : npt.NDArray or None
            an ndarray of the form ``C`` [``N samples``], containing the predicted classes
            for ``X``. Returns `None` if no detectors are available for prediction.
        """
        # If there are no detectors, Returns None.
        if self._detectors is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, len(self._detectors[self.classes[0]][0].position)
        )

        # Initializes an empty array that will store the predictions.
        c = []
        # For each sample row in X.
        for line in X:
            class_found: bool
            _class_ = self.__compare_sample_to_detectors(line)
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
                    detectores = list(
                        map(lambda x: x.position, self._detectors[_class_])
                    )
                    average_distance[_class_] = np.average(
                        [self.__distance(detector, line) for detector in detectores]
                    )
                c.append(max(average_distance, key=average_distance.get))
        return np.array(c)

    def __checks_valid_detector(
        self, x_class: npt.NDArray, vector_x: npt.NDArray
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
        Validity : bool
            Returns whether the detector is valid or not.
        """
        # If any of the input arrays have zero size, Returns false.
        if np.size(x_class) == 0 or np.size(vector_x) == 0:
            return False
        # If self.k > 1, uses the k nearest neighbors (kNN); otherwise, checks the detector
        # without considering kNN.
        if self.k > 1:
            knn_list = []
            for x in x_class:
                # Calculates the distance between the two vectors and adds it to the kNN list if
                # the distance is smaller than the largest distance in the list.
                knn_list = self.__compare_knearest_neighbors_list(
                    knn_list, self.__distance(x, vector_x)
                )
            # If the average of the distances in the kNN list is less than the radius, Returns true.
            distance_mean = np.mean(knn_list)
            if self.algorithm == "V-detector":
                return self.__detector_is_valid_to_vdetector(distance_mean, vector_x)
            if distance_mean > (self.r + self.r_s):
                return True
        else:
            if self.algorithm == "V-detector":
                distance = min_distance_to_class_vectors(
                    x_class, vector_x, get_metric_code(self.metric), self.p
                )
                return self.__detector_is_valid_to_vdetector(distance, vector_x)

            # Calculates the distance between the vectors; if not it is less than or equal to
            # the radius plus the sample's radius, sets the validity of the detector to
            # true.
            threshold: float = self.r + self.r_s
            if check_detector_rnsa_validity(
                x_class, vector_x, threshold, get_metric_code(self.metric), self.p
            ):
                return True  # Detector is valid!

        return False  # Detector is not valid!

    def __compare_knearest_neighbors_list(
        self, knn: npt.NDArray, distance: float
    ) -> npt.NDArray:
        """
        Compare the k-nearest neighbor distance at position k=1 in the list knn.

        If the distance of the new sample is less, replace it and sort in ascending order.

        Parameters
        ----------
        knn : npt.NDArray
            List of k-nearest neighbor distances.
        distance : float
            Distance to check.

        Returns
        -------
        knn : npt.NDArray
            Updated and sorted nearest neighbor list.
        """
        # If the number of distances in kNN is less than k, adds the distance.
        if len(knn) < self.k:
            knn = np.append(knn, distance)
            knn.sort()
            return knn

        # Otherwise, add the distance if the new distance is smaller than the largest
        # distance in the list.
        if knn[self.k - 1] > distance:
            knn[self.k - 1] = distance
            knn.sort()

        return knn

    def __compare_sample_to_detectors(self, line: npt.NDArray) -> Optional[str]:
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
        # List to store the classes and the average distance between the detectors and the sample.
        possible_classes = []
        for _class_ in self.classes:
            # Variable to indicate if the class was found with the detectors.
            class_found: bool = True
            sum_distance = 0
            for detector in self._detectors[_class_]:
                distance = self.__distance(detector.position, line)
                sum_distance += distance
                if self.algorithm == "V-detector":
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

    def __distance(self, u: npt.NDArray, v: npt.NDArray) -> float:
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

    def __detector_is_valid_to_vdetector(
        self, distance: float, vector_x: npt.NDArray
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

        return (True, new_detector_r)


class BNSA(BaseNSA):
    """BNSA (Binary Negative Selection Algorithm).
    
    Class is for classification and identification purposes of anomalies through the self and not
    self method.

    Attributes
    ----------
    N : int, default=100
        Number of detectors.
    aff_thresh : float, default=0.1
        The variable represents the percentage of similarity between the T cell and the own
        samples. The default value is 10% (0.1), while a value of 1.0 represents 100% similarity.
    max_discards : int, default=1000
        This parameter indicates the maximum number of detector discards in sequence, which aims
        to avoid a possible infinite loop if a radius is defined that it is not possible to
        generate non-self detectors. Defaults to ``1000``.
    seed : Optional[int], default=None
             Seed for the random generation of values in the detectors. Defaults to
        ``None``.
    no_label_sample_selection : str
        Method for selecting labels for samples designated as non-self by all detectors.
        Available method types:

        - max_average_difference - Selects the class with the highest average difference among the
        detectors.

        - max_nearest_difference - Selects the class with the highest difference between the
        nearest and farthest detector from the sample.
    """

    def __init__(
        self,
        N: int = 100,
        aff_thresh: float = 0.1,
        max_discards: int = 1000,
        seed: int = None,
        no_label_sample_selection: Literal[
            "max_average_difference", "max_nearest_difference"
        ] = "max_average_difference",
    ):
        super().__init__()

        self.N: int = sanitize_param(N, 100, lambda x: x > 0)
        self.aff_thresh: float = sanitize_param(aff_thresh, 0.1, lambda x: 0 < x < 1)
        self.max_discards: float = sanitize_param(max_discards, 1000, lambda x: x > 0)

        self.seed = sanitize_seed(seed)

        if self.seed is not None:
            np.random.seed(seed)

        self.no_label_sample_selection: float = sanitize_param(
            no_label_sample_selection,
            "max_average_difference",
            lambda x: x == "nearest_difference",
        )

        self.classes: npt.NDArray = None
        self._detectors: Optional[dict] = None
        self._detectors_stack: npt.NDArray = None

    @property
    def detectors(self) -> Dict[str, npt.NDArray[np.bool_]]:
        """Returns the trained detectors, organized by class."""
        return self._detectors

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """Training according to X and y, using the method negative selection method.

        Parameters
        ----------
        X : npt.NDArray
            Training array, containing the samples and their characteristics, [``N samples`` (
            rows)][``N features`` (columns)].
        y : npt.NDArray
            Array of target classes of ``X`` with [``N samples`` (lines)].
        verbose : bool, default=True
            Feedback from detector generation to the user.

        Returns
        -------
        self : BNSA
             Returns the instance it self.
        """
        super()._check_and_raise_exceptions_fit(X, y, "BNSA")

        # Converts the entire array X to boolean
        X = X.astype(np.bool_)

        # Identifying the possible classes within the output array `y`.
        self.classes = np.unique(y)
        # Dictionary that will store detectors with classes as keys.
        list_detectors_by_class = {}
        # Separates the classes for training.
        sample_index: dict = self._slice_index_list_by_class(y)
        # Progress bar for generating all detectors.
        if verbose:
            progress = tqdm(
                total=int(self.N * (len(self.classes))),
                bar_format="{desc} ┇{bar}┇ {n}/{total} detectors",
                postfix="\n",
            )

        for _class_ in self.classes:
            # Initializes the empty set that will contain the valid detectors.
            valid_detectors_set: list = []
            discard_count: int = 0
            # Updating the progress bar with the current class the algorithm is processing.
            if verbose:
                progress.set_description_str(
                    f"Generating the detectors for the {_class_} class:"
                )
            x_class = X[sample_index[_class_]]
            while len(valid_detectors_set) < self.N:
                # Generates a candidate detector vector randomly with values 0 and 1.
                vector_x = np.random.randint(0, 2, size=X.shape[1]).astype(np.bool_)
                # If the detector is valid, add it to the list of valid detectors.
                if check_detector_bnsa_validity(x_class, vector_x, self.aff_thresh):
                    discard_count = 0
                    valid_detectors_set.append(vector_x)
                    if verbose:
                        progress.update(1)
                else:
                    discard_count += 1
                    if discard_count == self.max_discards:
                        raise MaxDiscardsReachedError(_class_)

            # Add detectors to the dictionary with classes as keys.
            list_detectors_by_class[_class_] = np.array(valid_detectors_set)

        # Notify the completion of detector generation for the classes.
        if verbose:
            progress.set_description(
                f"\033[92m✔ Non-self detectors for classes ({', '.join(map(str, self.classes))}) "
                f"successfully generated\033[0m"
            )
        # Saves the found detectors in the attribute for the class detectors.
        self._detectors = list_detectors_by_class
        self._detectors_stack = np.array(
            [np.stack(self._detectors[class_name]) for class_name in self.classes]
        )
        return self

    def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]:
        """Prediction of classes based on detectors created after training.

        Parameters
        ----------
        X : npt.NDArray
            Array with input samples with [``N_samples`` (Lines)] and [``N_characteristics``(
            Columns)]

        Returns
        -------
        c : Optional[npt.NDArray]
            an ndarray of the form ``C`` [``N samples``], containing the predicted classes for
            ``X``. Returns``None``: If there are no detectors for the prediction.
        """
        # If there are no detectors, Returns None.
        if self._detectors is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, len(self._detectors[self.classes[0]][0]), "BNSA"
        )

        # Converts the entire array X to boolean.
        if X.dtype != bool:
            X = X.astype(bool)

        # Initializes an empty array that will store the predictions.
        c = []
        # For each sample row in X.
        for line in X:
            class_found: bool = True
            # Class prediction based on detectors
            class_index = bnsa_class_prediction(
                line, self._detectors_stack, self.aff_thresh
            )
            # If belonging to one or more classes, adds the class with the greatest
            # average distance
            if class_index > -1:
                c.append(self.classes[class_index])
                class_found = True
            else:
                class_found = False

            # If there is only one class and the sample is not classified, sets the
            # output as non-self.
            if not class_found and len(self.classes) == 1:
                c.append("non-self")
            # If the class cannot be identified by the detectors
            elif not class_found:
                self.__assign_class_to_non_self_sample(line, c)

        return np.array(c)

    def __assign_class_to_non_self_sample(self, line: npt.NDArray, c: list):
        """Determine the class of a sample when all detectors classify it as "non-self".
        
        Classification is performed using the ``max_average_difference`` and
        ``max_nearest_difference`` methods.

        Parameters
        ----------
        line : list
            Sample to be classified.
        c : list
            List of predictions to be updated with the new classification.
        """
        class_differences: dict = {}
        for _class_ in self.classes:
            distances = np.sum(line != self._detectors[_class_]) / self.N
            # Assign the label to the class with the greatest distance from
            # the nearest detector.
            if self.no_label_sample_selection == "nearest_difference":
                class_differences[_class_] = distances.min()
            # Or based on the greatest distance from the average distances of the detectors.
            else:
                class_differences[_class_] = distances.sum() / self.N

        c.append(max(class_differences, key=class_differences.get))
