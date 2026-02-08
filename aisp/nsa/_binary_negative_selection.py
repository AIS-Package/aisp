"""Negative Selection Algorithm."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from ._base import check_detector_bnsa_validity, bnsa_class_prediction
from ..base import BaseClassifier
from ..exceptions import MaxDiscardsReachedError, ModelNotFittedError
from ..utils.sanitizers import sanitize_seed, sanitize_param
from ..utils.validation import (
    check_array_type,
    check_shape_match,
    check_binary_array,
    check_feature_dimension,
)


class BNSA(BaseClassifier):
    """Binary Negative Selection Algorithm (BNSA).

    Algorithm for classification and anomaly detection Based on self or not self
    discrimination, inspired by Negative Selection Algorithm.

    Parameters
    ----------
    N : int, default=100
        Number of detectors.
    aff_thresh : float, default=0.1
        The variable represents the percentage of similarity between the T cell and the own
        samples. The default value is 10% (0.1), while a value of 1.0 represents 100% similarity.

        Warning
            High values may prevent the generation of valid non-self detectors.
    max_discards : int, default=1000
        This parameter indicates the maximum number of detector discards in sequence, which aims
        to avoid a possible infinite loop if a radius is defined that it is not possible to
        generate non-self detectors.
    seed : Optional[int], default=None
         Seed for the random generation of values in the detectors.
    no_label_sample_selection : str, default="max_average_difference"
        Method for selecting labels for samples designated as non-self by all detectors.
        Available method types:

        - max_average_difference - Selects the class with the highest average difference among the
        detectors.

        - max_nearest_difference - Selects the class with the highest difference between the
        nearest and farthest detector from the sample.

    Notes
    -----
    The **Binary Negative Selection Algorithm (BNSA)** is based on the original proposal by
    Forrest et al. (1994) [1], originally developed for computer security. In the adaptation, the
    algorithm use bits arrays, and it has support for multiclass classification.

    References
    ----------
    .. [1] S. Forrest, A. S. Perelson, L. Allen and R. Cherukuri, "Self-nonself discrimination in
        a computer," Proceedings of 1994 IEEE Computer Society Symposium on Research in Security
        and Privacy, Oakland, CA, USA, 1994, pp. 202-212,
        doi: https://dx.doi.org/10.1109/RISP.1994.296580.

    Examples
    --------
    >>> from aisp.nsa import BNSA
    >>> # Binary 'self' samples
    >>> x_train  = [
    ...     [0, 0, 1, 0, 1],
    ...     [0, 1, 1, 0, 1],
    ...     [0, 1, 0, 1, 0],
    ...     [0, 0, 1, 0, 1],
    ...     [0, 1, 1, 0, 1],
    ...     [0, 1, 0, 1, 0]
    ... ]
    >>> y_train = ['self', 'self', 'self', 'self', 'self', 'self']
    >>> bnsa = BNSA(aff_thresh=0.55, seed=1)
    >>> bnsa = bnsa.fit(x_train, y_train, verbose=False)
    >>> # samples for testing
    >>> x_test = [
    ...     [1, 1, 1, 1, 1], # Sample of Anomaly
    ...     [0, 1, 0, 1, 0], # self sample
    ... ]
    >>> y_prev = bnsa.predict(X=x_test)
    >>> print(y_prev)
    ['non-self' 'self']
    """

    def __init__(
        self,
        N: int = 100,
        aff_thresh: float = 0.1,
        max_discards: int = 1000,
        seed: Optional[int] = None,
        no_label_sample_selection: Literal[
            "max_average_difference", "max_nearest_difference"
        ] = "max_average_difference",
    ):
        self.N: int = sanitize_param(N, 100, lambda x: x > 0)
        self.aff_thresh: float = sanitize_param(aff_thresh, 0.1, lambda x: 0 < x < 1)
        self.max_discards: int = sanitize_param(max_discards, 1000, lambda x: x > 0)

        self.seed: Optional[int] = sanitize_seed(seed)

        if self.seed is not None:
            np.random.seed(seed)

        self.no_label_sample_selection: str = sanitize_param(
            no_label_sample_selection,
            "max_average_difference",
            lambda x: x == "nearest_difference",
        )

        self.classes: Optional[npt.NDArray] = None
        self._detectors: Optional[dict] = None
        self._detectors_stack: Optional[npt.NDArray] = None

    @property
    def detectors(self) -> Optional[Dict[str | int, npt.NDArray[np.bool_]]]:
        """Return the trained detectors, organized by class."""
        return self._detectors

    def fit(
        self,
        X: Union[npt.NDArray, list],
        y: Union[npt.NDArray, list],
        verbose: bool = True,
    ) -> BNSA:
        """Training according to X and y, using the method negative selection method.

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Training array, containing the samples and their characteristics.
            Shape: (``n_samples, n_features``)
        y : Union[npt.NDArray, list]
            Array of target classes of ``X`` with ``n_samples`` (lines).
        verbose : bool, default=True
            Feedback from detector generation to the user.

        Raises
        ------
        TypeError
            If X or y are not ndarrays or have incompatible shapes.
        ValueError
            If the array contains values other than 0 and 1.
        MaxDiscardsReachedError
            The maximum number of detector discards was reached during maturation. Check the
            defined radius value and consider reducing it.

        Returns
        -------
        self : BNSA
             Returns the instance itself.
        """
        X = check_array_type(X)
        y = check_array_type(y, "y")
        check_shape_match(X, y)
        check_binary_array(X)

        # Converts the entire array X to boolean
        X = X.astype(np.bool_)
        self._n_features = X.shape[1]
        # Identifying the possible classes within the output array `y`.
        self.classes = np.unique(y)
        # Dictionary that will store detectors with classes as keys.
        list_detectors_by_class: dict = {}
        # Separates the classes for training.
        sample_index: dict = self._slice_index_list_by_class(y)
        # Progress bar for generating all detectors.

        progress = tqdm(
            total=int(self.N * (len(self.classes))),
            bar_format="{desc} ┇{bar}┇ {n}/{total} detectors",
            postfix="\n",
            disable=not verbose,
        )

        for _class_ in self.classes:
            # Initializes the empty set that will contain the valid detectors.
            valid_detectors_set: list = []
            discard_count: int = 0
            # Updating the progress bar with the current class the algorithm is processing.
            progress.set_description_str(
                f"Generating the detectors for the {_class_} class:"
            )
            x_class = X[sample_index[_class_]]
            while len(valid_detectors_set) < self.N:
                # Generates a candidate detector vector randomly with values 0 and 1.
                vector_x = np.random.randint(0, 2, size=(self._n_features,)).astype(np.bool_)
                # If the detector is valid, add it to the list of valid detectors.
                if check_detector_bnsa_validity(x_class, vector_x, self.aff_thresh):
                    discard_count = 0
                    valid_detectors_set.append(vector_x)
                    progress.update()
                else:
                    discard_count += 1
                    if discard_count == self.max_discards:
                        raise MaxDiscardsReachedError(_class_)

            # Add detectors to the dictionary with classes as keys.
            list_detectors_by_class[_class_] = np.array(valid_detectors_set)

        # Notify the completion of detector generation for the classes.
        progress.set_description(
            f"\033[92m✔ Non-self detectors for classes ({', '.join(map(str, self.classes))}) "
            f"successfully generated\033[0m"
        )
        progress.close()
        # Saves the found detectors in the attribute for the class detectors.
        self._detectors = list_detectors_by_class
        self._detectors_stack = np.array(
            [np.stack(self._detectors[class_name]) for class_name in self.classes]
        )

        return self

    def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
        """Prediction of classes based on detectors created after training.

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Array with input samples with Shape: (``n_samples, n_features``)

        Raises
        ------
        TypeError
            If X is not a ndarray or list.
        ValueError
            If the array contains values other than 0 and 1.
        FeatureDimensionMismatch
            If the number of features in X does not match the expected number.
        ModelNotFittedError
            If the mode has not yet been adjusted and does not have defined detectors or
            classes, it is not able to predictions

        Returns
        -------
        c : npt.NDArray
            A ndarray of the form ``C`` (``n_samples``), containing the predicted classes for
            ``X``.
        """
        if (
            self._detectors is None
            or self._detectors_stack is None
            or self.classes is None
        ):
            raise ModelNotFittedError("BNSA")
        X = check_array_type(X)
        check_feature_dimension(X, self._n_features)
        check_binary_array(X)

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
                self._assign_class_to_non_self_sample(line, c)

        return np.array(c)

    def _assign_class_to_non_self_sample(self, line: npt.NDArray, c: list):
        """Determine the class of a sample when all detectors classify it as "non-self".

        Classification is performed using the ``max_average_difference`` and
        ``max_nearest_difference`` methods.

        Parameters
        ----------
        line : npt.NDArray
            Sample to be classified.
        c : list
            List of predictions to be updated with the new classification.

        Raises
        ------
        ValueError
            If detectors is not initialized.
        """
        if self._detectors is None or self.classes is None:
            raise ValueError("Detectors is not initialized.")

        class_differences: dict = {}
        for _class_ in self.classes:
            distances = np.mean(line != self._detectors[_class_], axis=1)
            # Assign the label to the class with the greatest distance from
            # the nearest detector.
            if self.no_label_sample_selection == "nearest_difference":
                class_differences[_class_] = distances.min()
            # Or based on the greatest distance from the average distances of the detectors.
            else:
                class_differences[_class_] = distances.sum() / self.N

        c.append(max(class_differences, key=class_differences.get))  # type: ignore
