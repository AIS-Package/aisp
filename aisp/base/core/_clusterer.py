"""Base class for clustering algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy.typing as npt

from ._base import Base


class BaseClusterer(ABC, Base):
    """Abstract base class for clustering algorithms.

    This class defines the core interface for clustering models. It enforces
    the implementation of the `fit` and `predict` methods in all derived classes,
    and provides a default implementation for `fit_predict` and `get_params`.
    """

    labels: Optional[npt.NDArray] = None

    @abstractmethod
    def fit(self, X: Union[npt.NDArray, list], verbose: bool = True) -> BaseClusterer:
        """
        Train the model using the input data X.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Input data used for training the model.
        verbose : bool, default=True
            Flag to enable or disable detailed output during training.

        Returns
        -------
        self : BaseClusterer
            Returns the instance of the class that implements this method.
        """

    @abstractmethod
    def predict(self, X: Union[npt.NDArray, list]) -> npt.NDArray:
        """
        Generate predictions based on the input data X.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Input data for which predictions will be generated.

        Returns
        -------
        predictions : npt.NDArray
            Predicted cluster labels for each input sample.
        """

    def fit_predict(self, X: Union[npt.NDArray, list], verbose: bool = True) -> npt.NDArray:
        """Fit the clustering model to the data and return cluster labels.

        This is a convenience method that combines `fit` and `predict`
        into a single call.

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Input data for which predictions will be generated.
        verbose : bool, default=True
            Flag to enable or disable detailed output during training.

        Returns
        -------
        predictions : npt.NDArray
            Predicted cluster labels for each input sample.
        """
        self.fit(X, verbose)
        return self.predict(X)
