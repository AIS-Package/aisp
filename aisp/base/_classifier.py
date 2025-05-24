"""Base class for classification algorithm."""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy.typing as npt

from ..utils import slice_index_list_by_class
from ..utils.metrics import accuracy_score


class BaseClassifier(ABC):
    """Base class for classification algorithms.

    Defines the abstract methods ``fit`` and ``predict``, and implements the ``score``,
    ``get_params`` method.
    """

    classes: Optional[Union[npt.NDArray, list]] = None

    @abstractmethod
    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
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

    def get_params(self, deep: bool = True) -> dict:  # pylint: disable=W0613
        """
        Return a dictionary with the object's main parameters.

        This method is required to ensure compatibility with scikit-learn functions.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
