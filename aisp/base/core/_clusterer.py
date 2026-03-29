"""Base class for clustering algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union
from warnings import warn

import numpy.typing as npt

from ._base import Base


class BaseClusterer(ABC, Base):
    """Abstract base class for clustering algorithms.

    This class defines the core interface for clustering models. It enforces
    the implementation of the `fit` and `predict` methods in all derived classes,
    and provides a default implementation for `fit_predict` and `get_params`.

    Attributes
    ----------
    labels : Optional[npt.NDArray]
        Labels for the clusters generated during model fitting.
    """

    labels: Optional[npt.NDArray] = None

    @property
    def classes(self) -> Optional[npt.NDArray]:
        """Deprecated alias kept for backward compatibility.

        Use `labels` instead of `classes`.
        """
        warn(
            "The `classes` attribute is deprecated and will be removed in future "
            "versions. Use labels instead.",
            FutureWarning,
            2
        )
        return self.labels

    @abstractmethod
    def fit(self, X: Union[npt.NDArray, list], verbose: bool = True) -> BaseClusterer:
        """
        Train the model using the input data X.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        X : Union[npt.NDArray, list]
            Training input samples. Each row corresponds to a samples and column to feature.
        verbose : bool, default=True
            If True, prints training progress information.

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
            Input samples. Must have the same number of features used during training.

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
            Training input samples. Each row corresponds to a samples and column to feature.
        verbose : bool, default=True
            If True, prints training progress information.

        Returns
        -------
        predictions : npt.NDArray
            Predicted cluster labels for each input sample.
        """
        self.fit(X, verbose)
        return self.predict(X)
