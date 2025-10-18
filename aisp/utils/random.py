"""Utility functions for random number generation and reproducibility."""

import random

import numpy as np
from numba import njit


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
