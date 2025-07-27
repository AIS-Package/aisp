"""Represents a memory B-cell."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from ..base.mutation import (
    clone_and_mutate_continuous,
    clone_and_mutate_binary,
    clone_and_mutate_ranged
)
from ..utils.types import FeatureType


@dataclass(slots=True)
class Cell:
    """
    Represents a memory B-cell.

    Attributes
    ----------
    vector : npt.NDArray
        A vector of cell features.
    """

    vector: np.ndarray

    def hyper_clonal_mutate(
        self,
        n: int,
        feature_type: FeatureType = "continuous-features",
        bounds: Optional[npt.NDArray[np.float64]] = None
    ) -> npt.NDArray:
        """
        Clones N features from a cell's features, generating a set of mutated vectors.

        Parameters
        ----------
        n : int
            Number of clones to be generated from mutations of the original cell.
        feature_type : Literal["binary-features", "continuous-features", "ranged-features"]
            Specifies the type of feature_type to use based on the nature of the input features
        bounds : np.ndarray
            Array (n_features, 2) with min and max per dimension.

        Returns
        -------
        npt.NDArray
            An array containing N mutated vectors from the original cell.
        """
        if feature_type == "binary-features":
            return clone_and_mutate_binary(self.vector, n)
        if feature_type == "ranged-features" and bounds is not None:
            clone_and_mutate_ranged(self.vector, n, bounds)
        return clone_and_mutate_continuous(self.vector, n)

    def __eq__(self, other):
        """Check if two cells are equal."""
        return np.array_equal(self.vector, other.vector)
