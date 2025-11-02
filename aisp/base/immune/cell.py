"""Representation of immune system cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt

from .mutation import (
    clone_and_mutate_binary,
    clone_and_mutate_ranged,
    clone_and_mutate_continuous
)
from ...utils.types import FeatureType


@dataclass(slots=True)
class Cell:
    """Represents a cell.

    Attributes
    ----------
    vector : np.ndarray
        A vector of cell features.
    """

    vector: np.ndarray

    def __eq__(self, other):
        """Check if two cells are equal."""
        if isinstance(other, Cell):
            return np.array_equal(self.vector, other.vector)
        if isinstance(other, (list, np.ndarray)):
            return np.array_equal(self.vector, other)

        return NotImplemented

    def __array__(self) -> np.ndarray:
        """Array interface to Numpy.

        Allows the instance to be treated as a np.ndarray by NumPy functions.

        Returns
        -------
        np.ndarray
            The cell's feature vector (self.vector).
        """
        return self.vector

    def __getitem__(self, item):
        """Get the index of the NumPy vector."""
        return self.vector[item]


@dataclass(slots=True, eq=False)
class BCell(Cell):
    """
    Represents a memory B-cell.

    Attributes
    ----------
    vector : npt.NDArray
        A vector of cell features.
    """

    def hyper_clonal_mutate(
        self,
        n: int,
        feature_type: FeatureType = "continuous-features",
        bounds: Optional[npt.NDArray[np.float64]] = None
    ) -> np.ndarray:
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
            clone_and_mutate_ranged(self.vector, n, bounds, np.float64(1.0))
        return clone_and_mutate_continuous(self.vector, n, np.float64(1.0))


@dataclass(slots=True)
class Antibody(Cell):
    """
    Represent an antibody.

    Attributes
    ----------
    vector : npt.NDArray
        A vector of cell features.
    affinity : np.floating
        Affinity value.
    """

    affinity: float

    def __lt__(self, other):
        """Compare this cell with another Antibody cell based on affinity."""
        if isinstance(other, Antibody):
            return self.affinity < other.affinity
        if isinstance(other, (float, int, np.floating)):
            return self.affinity < other
        return NotImplemented

    def __eq__(self, other):
        """Check if this cell has the same affinity as another cell."""
        if isinstance(other, Antibody):
            return self.affinity == other.affinity
        if isinstance(other, (float, int, np.floating)):
            return self.affinity == other
        return NotImplemented


@dataclass(slots=True)
class Detector:
    """
    Represents a non-self detector of the RNSA class.

    Attributes
    ----------
    position : npt.NDArray[np.float64]
        Detector feature vector.
    radius : float, optional
        Detector radius, used in the V-detector algorithm.
    """

    position: npt.NDArray[np.float64]
    radius: Optional[float] = None
