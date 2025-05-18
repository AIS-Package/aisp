"""Represents a memory B-cell."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..base.mutation import clone_and_mutate_continuous, clone_and_mutate_binary


@dataclass(slots=True)
class Cell:
    """Cell

    Represents a memory B-cell.

    Parameters:
    ----------
    * vector (``npt.NDArray``): A vector of cell features.
    """

    vector: np.ndarray

    def hyper_clonal_mutate(
        self,
        n: int,
        algorithm: Literal["continuous-features", "binary-features"] = "continuous-features"
    ) -> npt.NDArray:
        """
        Clones N features from a cell's features, generating a set of mutated vectors.

        Parameters
        ----------
        * n (``int``): Number of clones to be generated from mutations of the original cell.

        Returns
        ----------
        * npt.NDArray: An array containing N mutated vectors from the original cell.
        """
        if algorithm == "binary-features":
            return clone_and_mutate_binary(self.vector, n)
        return clone_and_mutate_continuous(self.vector, n)
