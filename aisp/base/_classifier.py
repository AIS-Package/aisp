"""Base class for classification algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy.typing as npt

from ._base import Base
from ..utils import slice_index_list_by_class
from ..utils.metrics import accuracy_score


class BaseClassifier(ABC, Base):
    """Abstract base class for classification algorithms.

    This class defines the core interface for classification models. It enforces the
    implementation of the ``fit`` and ``predict`` methods in all derived classes,
    and provides a default implementation of ``score`` and utility functions.
    """

    classes: Union[npt.NDArray, list] = []

    @abstractmethod
    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True) -> BaseClassifier:
        """
        Train the model using the input data X and corresponding labels y.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        X : npt.NDArray
            Input data used for training the model.
        y : npt.NDArray
            Corresponding labels or target values for the input data.
        verbose : bool, default=True
            Flag to enable or disable detailed output during training.

        Returns
        -------
        self : BaseClassifier
            Returns the instance of the class that implements this method.
        """

    @abstractmethod
    def predict(self, X) -> Optional[npt.NDArray]:
        """
        Generate predictions based on the input data X.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        X : npt.NDArray
            Input data for which predictions will be generated.

        Returns
        -------
        Predictions : Optional[npt.NDArray]
            Predicted values for each input sample, or ``None`` if the prediction fails.
        """

    def score(self, X: npt.NDArray, y: list) -> float:
        """
        Score function calculates forecast accuracy.

        Details
        -------
        This function performs the prediction of X and checks how many elements are equal
        between vector y and y_predicted. This function was added for compatibility with some
        scikit-learn functions.

        Parameters
        ----------
        X : np.ndarray
            Feature set with shape (n_samples, n_features).
        y : np.ndarray
            True values with shape (n_samples,).

        Returns
        -------
        accuracy : float
            The accuracy of the model.
        """
        if len(y) == 0:
            return 0
        y_pred = self.predict(X)

        if y_pred is None:
            return 0

        return accuracy_score(y, y_pred)

    def _slice_index_list_by_class(self, y: npt.NDArray) -> dict:
        """Separate the indices of the lines according to the output class.

        Loop through the sample array only in positions where the output matches the class
        being trained.

        Parameters
        ----------
        y : npt.NDArray
            Receives a y [``N sample``] array with the output classes of the ``X`` sample array.

        Returns
        -------
        dict: dict
            A dictionary with the list of array positions(``y``), with the classes as key.
        """
        return slice_index_list_by_class(self.classes, y)
