"""Base class for clustering algorithms."""

from abc import ABC, abstractmethod

import numpy.typing as npt


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms.

    This class defines the core interface for clustering models. It enforces
    the implementation of the `fit` and `predict` methods in all derived classes,
    and provides a default implementation for `fit_predict` and `get_params`.
    """

    @abstractmethod
    def fit(self, X: npt.NDArray, verbose: bool = True):
        """
        Train the model using the input data X.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        X : npt.NDArray
            Input data used for training the model.
        verbose : bool, default=True
            Flag to enable or disable detailed output during training.

        Returns
        -------
        self : BaseClassifier
            Returns the instance of the class that implements this method.
        """

    @abstractmethod
    def predict(self, X: npt.NDArray):
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

    def fit_predict(self, X):
        """Fit the clustering model to the data and return cluster labels.

        This is a convenience method that combines `fit` and `predict`
        into a single call.

        Parameters
        ----------
        X : npt.NDArray
            Input data for which predictions will be generated.

        Returns
        -------
        Predictions : Optional[npt.NDArray]
            Predicted values for each input sample, or ``None`` if the prediction fails.
        """
        self.fit(X)
        return self.predict(X)

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
