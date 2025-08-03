"""Base class for parameter introspection compatible with the scikit-learn API."""
import random

import numpy as np
from numba import njit


class Base:
    """
    Generic base class for models with a common interface.

    Provides the ``get_params`` and ``set_params`` method for compatibility with
    the scikit-learn API, allowing access to the model's public parameters.
    """

    def set_params(self, **params):
        """
        Set the parameters of the instance.

        This method is required to ensure compatibility with scikit-learn functions

        Parameters
        ----------
        **params
            set as attributes on the instance.

        Returns
        -------
        self
        """
        for key, value in params.items():
            if not key.startswith("_") and hasattr(self, key):
                setattr(self, key, value)
        return self

    def get_params(self, deep: bool = True) -> dict:  # pylint: disable=W0613
        """
        Return a dictionary with the object's main parameters.

        This method is required to ensure compatibility with scikit-learn functions.

        Returns
        -------
        dict
            Dictionary containing the object's attributes that do not start with "_".
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }


@njit(cache=True)
def set_seed_numba(seed: int):
    """
    Set the seed for random numbers used by functions compiled with Numba.

    Parameters
    ----------
    seed : int
        Integer value used to initialize Numba's random number generator.
    """
    np.random.seed(seed)
    random.seed(seed)
