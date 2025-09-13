"""Base Classes and Core Utilities.

This module provides the fundamental classes and utilities that serve as the basis for all
Artificial Immune Systems algorithms implemented in the AISP package.

Classes
-------
BaseClassifier
    Abstract Base Class for Classification Algorithms.
BaseClusterer
    Abstract Base Class for Clustering Algorithms.
BaseOptimizer
    Abstract Base Class for optimization algorithms.

Functions
---------
set_seed_numba
    Set Random Seed for Numba JIT Compilation.

"""

from ._base import set_seed_numba
from ._classifier import BaseClassifier
from ._clusterer import BaseClusterer
from ._optimizer import BaseOptimizer

__all__ = ['BaseClassifier', 'BaseClusterer', 'BaseOptimizer', 'set_seed_numba']
