"""Base class for classification algorithm."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy.typing as npt

from ..utils.metrics import accuracy_score


class BaseClassifier(ABC):
    """
    Base class for classification algorithms, defining the abstract methods ``fit`` and ``predict``,
    and implementing the ``get_params`` method.
    """

    @abstractmethod
    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        Function to train the model using the input data ``X`` and corresponding labels ``y``.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        * X (``npt.NDArray``): Input data used for training the model, previously normalized to the
            range [0, 1].
        * y (``npt.NDArray``): Corresponding labels or target values for the input data.
        * verbose (``bool``, optional): Flag to enable or disable detailed output during training.
            Default is ``True``.

        Returns
        ----------
        * self: Returns the instance of the class that implements this method.
        """

    @abstractmethod
    def predict(self, X) -> Optional[npt.NDArray]:
        """
        Function to generate predictions based on the input data ``X``.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        * X (``npt.NDArray``): Input data for which predictions will be generated.

        Returns
        ----------
        * Predictions (``Optional[npt.NDArray]``): Predicted values for each input sample, or
            ``None`` if the prediction fails.
        """

    def score(self, X: npt.NDArray, y: list) -> float:
        """
        Score function calculates forecast accuracy.

        Details
        ----------
        This function performs the prediction of X and checks how many elements are equal
        between vector y and y_predicted. This function was added for compatibility with some
        scikit-learn functions.

        Parameters
        ----------
        * X (``np.ndarray``):
            Feature set with shape (n_samples, n_features).
        * y (``np.ndarray``):
            True values with shape (n_samples,).

        Returns
        ----------
        * accuracy (``float``): The accuracy of the model.
        """
        if len(y) == 0:
            return 0
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep: bool = True) -> dict:  # pylint: disable=W0613
        """
        The get_params function Returns a dictionary with the object's main parameters.

        This function is required to ensure compatibility with scikit-learn functions.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
