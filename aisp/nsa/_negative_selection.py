"""Negative Selection Algorithm."""

from collections import namedtuple
from typing import Dict, Literal, Optional, Union
from tqdm import tqdm

import numpy as np
import numpy.typing as npt

from ._detectors_checkers import check_detector_bnsa_validity
from ..exceptions import MaxDiscardsReachedError
from ..utils import slice_index_list_by_class
from ..utils.sanitizers import sanitize_seed, sanitize_choice, sanitize_param
from ._base import Base


class RNSA(Base):
    """
    The ``RNSA`` (Real-Valued Negative Selection Algorithm) class is for classification and 
    identification purposes. of anomalies through the self and not self method.

    Parameters
    ----------
    * N (``int``): Number of detectors. Defaults to ``100``.
    * r (``float``): Radius of the detector. Defaults to ``0.05``.
    * r_s (``float``): rₛ Radius of the ``X`` own samples. Defaults to ``0.0001``.
    * k (``int``): Number of neighbors near the randomly generated detectors to perform the
        distance average calculation. Defaults to ``1``.
    * metric (``str``): Way to calculate the distance between the detector and the sample:
            + ``'Euclidean'`` ➜ The calculation of the distance is given by the expression:
                √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
            + ``'minkowski'`` ➜ The calculation of the distance is given by the expression:
                ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
            + ``'manhattan'`` ➜ The calculation of the distance is given by the expression:
                ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) .

            Defaults to ``'euclidean'``.
    * max_discards (``int``): This parameter indicates the maximum number of consecutive
        detector discards, aimed at preventing a possible infinite loop in case a radius
        is defined that cannot generate non-self detectors. Defaults to ``1000``.
    * seed (``int``): Seed for the random generation of values in the detectors. Defaults to
        ``None``.
    * algorithm(``str``), Set the algorithm version:
        * ``'default-NSA'``: Default algorithm with fixed radius.
        * ``'V-detector'``: This algorithm is based on the article \
            [Real-Valued Negative Selection Algorithm with Variable-Sized Detectors][2], \
            by Ji, Z., Dasgupta, D. (2004), and uses a variable radius for anomaly \
            detection in feature spaces.

        Defaults to ``'default-NSA'``.

    * ``**kwargs``:
        - non_self_label (``str``): This variable stores the label that will be assigned \
            when the data has only one output class, and the sample is classified as not \
            belonging to that class. Defaults to ``'non-self'``.
        - cell_bounds (``bool``): If set to ``True``, this option limits the generation \
            of detectors to the space within the plane between 0 and 1. This means that \
            any detector whose radius exceeds this limit is discarded, this variable is \
            only used in the ``V-detector`` algorithm. Defaults to ``False``.
        - p (``float``): This parameter stores the value of ``p`` used in the Minkowski \
            distance. The default is ``2``, which represents normalized Euclidean distance.\
            Different values of p lead to different variants of the [Minkowski Distance][1].

    Notes
    ----------
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
        **kwargs: Dict[str, Union[bool, str, float]]
    ):
        super().__init__(metric)

        self.metric = sanitize_choice(
            metric,
            ["manhattan", "minkowski"],
            "euclidean"
        )
        self.seed = sanitize_seed(seed)
        if self.seed is not None:
            np.random.seed(seed)
        self.k: int = sanitize_param(k, 1, lambda x: x > 1)
        self.N: int = sanitize_param(N, 100, lambda x: x >= 1)
        self.r: float = sanitize_param(r, 0.05, lambda x: x > 0)
        self.r_s: float = sanitize_param(r_s, 0.0001, lambda x: x > 0)

        if algorithm == "V-detector":
            self._detector = namedtuple("Detector", ["position", "radius"])
            self._algorithm: str = algorithm
        else:
            self._detector = namedtuple("Detector", "position")
            self._algorithm: str = "default-NSA"

        self.max_discards: int = sanitize_param(max_discards, 1000, lambda x: x > 0)

        # Retrieves the variables from kwargs.
        self.p: float = kwargs.get("p", 2)
        self._cell_bounds: bool = kwargs.get("cell_bounds", False)
        self.non_self_label: str = kwargs.get("non_self_label", "non-self")

        # Initializes the other class variables as None.
        self.detectors: Union[dict, None] = None
        self.classes: npt.NDArray = None

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        The function ``fit(...)``, performs the training according to ``X`` and ``y``, using the 
        method negative selection method(``NegativeSelect``).

        Parameters
        ----------
        * X (``npt.NDArray``): Training array, containing the samples and their \
            characteristics, [``N samples`` (rows)][``N features`` (columns)].
        * y (``npt.NDArray``): Array of target classes of ``X`` with [``N samples`` (lines)].
            verbose (``bool``): Feedback from detector generation to the user.
        
        Returns
        ----------
        * (``self``): Returns the instance itself.
        """
        progress = None
        super()._check_and_raise_exceptions_fit(X, y)

        # Identifying the possible classes within the output array `y`.
        self.classes = np.unique(y)
        # Dictionary that will store detectors with classes as keys.
        list_detectors_by_class = {}
        # Separates the classes for training.
        sample_index = self.__slice_index_list_by_class(y)
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
            # Indicating which class the algorithm is currently processing for the progress bar.
            if verbose:
                progress.set_description_str(
                    f"Generating the detectors for the {_class_} class:"
                )
            while len(valid_detectors_set) < self.N:
                # Generates a candidate detector vector randomly with values between 0 and 1.
                vector_x = np.random.random_sample(size=X.shape[1])
                # Checks the validity of the detector for non-self with respect to the class samples
                valid_detector = self.__checks_valid_detector(X[sample_index[_class_]], vector_x)

                # If the detector is valid, add it to the list of valid detectors.
                if valid_detector is not False:
                    discard_count = 0

                    if self._algorithm == "V-detector":
                        valid_detectors_set.append(
                            self._detector(vector_x, valid_detector[1])
                        )
                    else:
                        valid_detectors_set.append(self._detector(vector_x))

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
        self.detectors = list_detectors_by_class
        return self

    def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]:
        """
        Function to perform the prediction of classes based on detectors
        created after training.

        Parameters
        ----------
        * X (``npt.NDArray``)
            Array with input samples with [``N samples`` (Lines)] and
            [``N characteristics``(Columns)]

        Returns
        ----------
        * C (``npt.NDArray``)
            an ndarray of the form ``C`` [``N samples``], containing the predicted classes
            for ``X``.
        * ``None``
            If there are no detectors for the prediction.
        """
        # If there are no detectors, Returns None.
        if self.detectors is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, len(self.detectors[self.classes[0]][0].position)
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
                        map(lambda x: x.position, self.detectors[_class_])
                    )
                    average_distance[_class_] = np.average(
                        [self.__distance(detector, line) for detector in detectores]
                    )
                c.append(max(average_distance, key=average_distance.get))
        return np.array(c)

    def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
        """
        The function ``__slice_index_list_by_class(...)``, separates the indices of the lines 
        according to the output class, to loop through the sample array, only in positions where 
        the output is the class being trained.

        Parameters
        ----------
        * y (npt.NDArray)
            Receives a ``y``[``N sample``] array with the output classes of the \
            ``X`` sample array.

        Returns
        ----------
        * dict: A dictionary with the list of array positions(``y``), with the classes as key.
        """
        return slice_index_list_by_class(self.classes, y)

    def __checks_valid_detector(
        self,
        x_class: npt.NDArray = None,
        vector_x: npt.NDArray = None
    ) -> Union[bool, tuple[bool, float]]:
        """
        Function to check if the detector has a valid non-proper ``r`` radius for the class.

        Parameters
        ----------
        * x_class (``npt.NDArray``)
            Array ``x_class`` with the samples per class.
        * vector_x (``npt.NDArray``)
            Randomly generated vector x candidate detector with values between[0, 1].

        Returns
        ----------
        * Validity (``bool``): Returns whether the detector is valid or not.
        """
        # If any of the input arrays have zero size, Returns false.
        if (np.size(x_class) == 0 or np.size(vector_x) == 0):
            return False
        # If self.k > 1, uses the k nearest neighbors (kNN); otherwise, checks the detector
        # without considering kNN.
        if self.k > 1:
            knn_list = np.empty(shape=0)
            for x in x_class:
                # Calculates the distance between the two vectors and adds it to the kNN list if
                # the distance is smaller than the largest distance in the list.
                knn_list = self.__compare_knearest_neighbors_list(
                    knn_list, self.__distance(x, vector_x)
                )
            # If the average of the distances in the kNN list is less than the radius, Returns true.
            distance_mean = np.mean(knn_list)
            if self._algorithm == "V-detector":
                return self.__detector_is_valid_to_vdetector(distance_mean, vector_x)
            if distance_mean > (self.r + self.r_s):
                return True
        else:
            distance: Union[float, None] = None
            if self._algorithm == "V-detector":
                distance = min(
                    self.__distance(x, vector_x) for x in x_class
                )
                return self.__detector_is_valid_to_vdetector(distance, vector_x)

            # Calculates the distance between the vectors; if not it is less than or equal to
            # the radius plus the sample's radius, sets the validity of the detector to
            # true.
            threshold: float = self.r + self.r_s
            if all(self.__distance(x, vector_x) > threshold for x in x_class):
                return True # Detector is valid!

        return False  # Detector is not valid!

    def __compare_knearest_neighbors_list(
        self,
        knn: npt.NDArray,
        distance: float
    ) -> npt.NDArray:
        """
        Compares the k-nearest neighbor distance at position ``k-1`` in the list ``knn``,
        if the distance of the new sample is less, replace it and sort in ascending order.


        Parameters
        ----------
        * knn (``npt.NDArray``)
            List of k-nearest neighbor distances.
        * distance (``float``)
            Distance to check.

        Returns
        ----------
        * ``npt.NDArray``: Updated and sorted nearest neighbor list.
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

    def __compare_sample_to_detectors(self, line: npt.NDArray):
        """
        Function to compare a sample with the detectors, verifying if the sample is proper.

        Parameters
        ----------
        * line (``npt.NDArray``): vector with N-features

        Returns
        ----------
        * Returns the predicted class with the detectors or None if the sample does not qualify
        for any class.
        """
        # List to store the classes and the average distance between the detectors and the sample.
        possible_classes = []
        for _class_ in self.classes:
            # Variable to indicate if the class was found with the detectors.
            class_found: bool = True
            sum_distance = 0
            for detector in self.detectors[_class_]:
                distance = self.__distance(detector.position, line)
                sum_distance += distance
                if self._algorithm == "V-detector":
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
        Function to calculate the distance between two points by the chosen ``metric``.

        Parameters
        ----------
        * u (``npt.NDArray``): Coordinates of the first point.
        * v (``npt.NDArray``): Coordinates of the second point.

        Returns
        ----------
        * Distance (``float``): between the two points.
        """
        return super()._distance(u, v)

    def __detector_is_valid_to_vdetector(
        self,
        distance: float,
        vector_x: npt.NDArray
    ) -> Union[bool, tuple[bool, float]]:
        """
        Check if the distance between the detector and the samples, minus the radius of the samples,
        is greater than the minimum radius.

        Parameters
        ----------
        * distance (``float``): minimum distance calculated between all samples.
        * vector_x (``numpy.ndarray``): randomly generated candidate detector vector x with
            values between 0 and 1.

        Returns
        ----------
        * ``False`` if the calculated radius is smaller than the minimum distance or exceeds the
            edge of the space, if this option is enabled.
        * ``True`` and the distance minus the radius of the samples, if the radius is valid.`
        """
        new_detector_r = float(distance - self.r_s)
        if self.r >= new_detector_r:
            return False

        # If _cell_bounds is True, considers the detector to be within the plane bounds.
        if self._cell_bounds:
            for p in vector_x:
                if (p - new_detector_r) < 0 or (p + new_detector_r) > 1:
                    return False

        return (True, new_detector_r)

    def get_params(self, deep: bool = True) -> dict:  # pylint: disable=W0613
        """
        The get_params function Returns a dictionary with the object's main parameters.
        """
        return {
            "N": self.N,
            "r": self.r,
            "k": self.k,
            "metric": self.metric,
            "seed": self.seed,
            "algorithm": self._algorithm,
            "r_s": self.r_s,
            "cell_bounds": self._cell_bounds,
            "p": self.p,
        }


class BNSA(Base):
    """
    The ``BNSA`` (Binary Negative Selection Algorithm) class is for classification and 
    identification purposes of anomalies through the self and not self method.

    Parameters
    ----------
    * N (``int``): Number of detectors. Defaults to ``100``.
    * aff_thresh (``float``): The variable represents the percentage of similarity
        between the T cell and the own samples. The default value is 10% (0.1), while a value of
        1.0 represents 100% similarity.
    * max_discards (``int``): This parameter indicates the maximum number of detector discards in
        sequence, which aims to avoid a possible infinite loop if a radius is defined that it is
        not possible to generate non-self detectors. Defaults to ``1000``.
    * seed (``int``): Seed for the random generation of values in the detectors. Defaults to
        ``None``.
    * no_label_sample_selection (``str``): Method for selecting labels for samples designated as
        non-self by all detectors. Available method types:
        - (``max_average_difference``): Selects the class with the highest average difference
            among the detectors.
        - (``max_nearest_difference``): Selects the class with the highest difference between
            the nearest and farthest detector from the sample.
    """

    def __init__(
        self,
        N: int = 100,
        aff_thresh: float = 0.1,
        max_discards: int = 1000,
        seed: int = None,
        no_label_sample_selection: Literal[
            "max_average_difference", "max_nearest_difference"
        ] = "max_average_difference"
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
            lambda x: x == "nearest_difference"
        )

        self.classes: npt.NDArray = None
        self.detectors: npt.NDArray = None

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        The function ``fit(...)``, performs the training according to ``X`` and ``y``, using the 
        method negative selection method(``NegativeSelect``).

        Parameters
        ----------
        * X (``npt.NDArray``):
            Training array, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].
        * y (``npt.NDArray``):
            Array of target classes of ``X`` with [``N samples`` (lines)].
            verbose (``bool``): Feedback from detector generation to the user.
        
        Returns
        ----------
        * (``self``): Returns the instance itself.
        """
        super()._check_and_raise_exceptions_fit(X, y, "BNSA")

        # Converts the entire array X to boolean
        X = X.astype(np.bool_)

        # Identifying the possible classes within the output array `y`.
        self.classes = np.unique(y)
        # Dictionary that will store detectors with classes as keys.
        list_detectors_by_class = {}
        # Separates the classes for training.
        sample_index: dict = self.__slice_index_list_by_class(y)
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
            list_detectors_by_class[_class_] = valid_detectors_set

        # Notify the completion of detector generation for the classes.
        if verbose:
            progress.set_description(
                f"\033[92m✔ Non-self detectors for classes ({', '.join(map(str, self.classes))}) "
                f"successfully generated\033[0m"
            )
        # Saves the found detectors in the attribute for the class detectors.
        self.detectors = list_detectors_by_class
        return self

    def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]:
        """
        Function to perform the prediction of classes based on detectors
        created after training.

        Parameters
        ----------
        * X (``npt.NDArray``): Array with input samples with [``N samples`` (Lines)] and
            [``N characteristics``(Columns)]

        Returns
        ----------
        * c (``npt.NDArray``): an ndarray of the form ``C`` [``N samples``],
            containing the predicted classes for ``X``
        * ``None``: If there are no detectors for the prediction.
        """
        # If there are no detectors, Returns None.
        if self.detectors is None:
            return None

        super()._check_and_raise_exceptions_predict(
            X, len(self.detectors[self.classes[0]][0]), "BNSA"
        )

        # Converts the entire array X to boolean.
        if X.dtype != bool:
            X = X.astype(bool)

        # Initializes an empty array that will store the predictions.
        c = []
        # For each sample row in X.
        for line in X:
            class_found: bool = True
            # List to store the possible classes to which the sample matches with self
            # when compared to the non-self detectors.
            possible_classes: list = []
            for _class_ in self.classes:
                similarity_sum: float = 0
                # Calculates the Hamming distance between the row and all detectors.
                distances = np.mean(np.not_equal(line, self.detectors[_class_]), axis=1)

                # Check if any distance is below or equal to the threshold.
                if np.any(distances <= self.aff_thresh):
                    class_found = False
                else:
                    similarity_sum = distances.sum()

                # If the sample passes through all detectors of a class, adds the class as a
                # possible prediction and its average similarity.
                if class_found:
                    possible_classes.append([_class_, similarity_sum / self.N])

            # If belonging to one or more classes, adds the class with the greatest
            # average distance
            if len(possible_classes) > 0:
                c.append(max(possible_classes, key=lambda x: x[1])[0])
                class_found = True
            else:
                class_found = False

            # If there is only one class and the sample is not classified, sets the
            # output as non-self.
            if not class_found and len(self.classes) == 1:
                c.append("non-self")
            # If the class cannot be identified by the detectors
            elif not class_found:
                c = self.__assign_class_to_non_self_sample(line, c)

        return np.array(c)

    def __assign_class_to_non_self_sample(self, line: npt.NDArray, c: list) -> npt.NDArray:
        """
        This function determines the class of a sample when all detectors classify it
        as "non-self". Classification is performed using the ``max_average_difference``
        and ``max_nearest_difference`` methods.

        Parameters
        ----------
        * line (list): Sample to be classified.
        * c (list): List of predictions to be updated with the new classification.

        Returns
        ----------
        * list: The list of predictions `c` updated with the class assigned to the sample.
        """
        class_differences: dict = {}
        for _class_ in self.classes:
            # Assign the label to the class with the greatest distance from
            # the nearest detector.
            if self.no_label_sample_selection == "nearest_difference":
                difference_min: float = np.mean(
                    np.not_equal(line, self.detectors[_class_]), axis=1
                ).min()
                class_differences[_class_] = difference_min
            # Or based on the greatest distance from the average distances of the detectors.
            else:
                difference_sum: float = np.mean(
                    np.not_equal(line, self.detectors[_class_]), axis=1
                ).sum()
                class_differences[_class_] = difference_sum / self.N

        c.append(max(class_differences, key=class_differences.get))
        return c

    def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
        """
        The function ``__slice_index_list_by_class(...)``, separates the indices of the lines 
        according to the output class, to loop through the sample array, only in positions where 
        the output is the class being trained.

        Parameters
        ----------
        * y (``npt.NDArray``):
            Receives a ``y``[``N sample``] array with the output classes of the ``X``
            sample array.

        Returns
        ----------
        * dict: A dictionary with the list of array positions(``y``), with the classes as key.
        """
        return slice_index_list_by_class(self.classes, y)

    def get_params(self, deep: bool = True) -> dict:  # pylint: disable=W0613
        """
        The get_params function Returns a dictionary with the object's main parameters.
        """
        return {
            "N": self.N,
            "aff_thresh": self.aff_thresh,
            "max_discards": self.max_discards,
            "seed": self.seed,
        }
