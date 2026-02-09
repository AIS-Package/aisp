"""ARB (Artificial recognition ball)."""
from typing import Optional

import numpy.typing as npt

from ..base.immune.cell import BCell


class _ARB(BCell):
    """ARB (Artificial recognition ball).

    Individual from the set of recognizing cells (ARB), inherits characteristics from a B-cell,
    adding resource consumption

    Parameters
    ----------
    vector : npt.NDArray
        A vector of cell features.
    stimulation : Optional[float], default=None
        The rate at which the cell stimulates antigens.
    """

    def __init__(
        self, vector: npt.NDArray, stimulation: Optional[float] = None
    ) -> None:
        super().__init__(vector)
        self.resource: float = 0.0
        if stimulation is not None:
            self.stimulation: float = stimulation

    def consume_resource(self, n_resource: float, amplified: float = 1) -> float:
        """
        Update the amount of resources available for an ARB after consumption.

        This function consumes the resources and returns the remaining amount of resources after
        consumption.

        Parameters
        ----------
        n_resource : float
            Amount of resources.
        amplified : float
            Amplifier for the resource consumption by the cell. It is multiplied by the cell's
            stimulus. The default value is 1.

        Returns
        -------
        n_resource : float
            The remaining amount of resources after consumption.
        """
        consumption = self.stimulation * amplified
        n_resource -= consumption
        if n_resource < 0:
            return 0

        self.resource = consumption
        return n_resource

    def to_cell(self) -> BCell:
        """Convert this _ARB into a pure BCell object."""
        return BCell(self.vector)
