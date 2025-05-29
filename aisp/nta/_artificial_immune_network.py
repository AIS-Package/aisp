"""Artificial Immune Network."""

from typing import Optional, Literal

import numpy as np
from numpy import typing as npt

from ._base import BaseAiNet
from ..utils.sanitizers import sanitize_param, sanitize_seed, sanitize_choice


class AiNet(BaseAiNet):
    """Artificial Immune Network
    """

    def __init__(
        self,
        k: int = 3,
        metric: Literal["manhattan", "minkowski", "euclidean"] = "euclidean",
        algorithm: Literal[
            "continuous-features", "binary-features"
        ] = "continuous-features",
        seed: int = None,
        **kwargs,
    ):
        self.k: int = sanitize_param(k, 3, lambda x: x > 3)
        self.seed: int = sanitize_seed(seed)
        if self.seed is not None:
            np.random.seed(self.seed)

        self.algorithm: Literal["continuous-features", "binary-features"] = (
            sanitize_param(
                algorithm, "continuous-features", lambda x: x == "binary-features"
            )
        )

        if algorithm == "binary-features":
            self.metric: str = "hamming"
        else:
            self.metric: str = sanitize_choice(
                metric, ["manhattan", "minkowski"], "euclidean"
            )

        self.p: np.float64 = np.float64(kwargs.get("p", 2.0))

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        Train the model using the input data X and corresponding labels y.

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

        return self

    def predict(self, X) -> Optional[npt.NDArray]:
        """
        Generate predictions based on the input data X.

        Parameters
        ----------
        X : npt.NDArray
            Input data for which predictions will be generated.

        Returns
        -------
        Predictions : Optional[npt.NDArray]
            Predicted values for each input sample, or ``None`` if the prediction fails.
        """

        return None
