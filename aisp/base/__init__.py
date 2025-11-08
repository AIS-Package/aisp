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

Module
------
immune
    Support Module for Artificial Immune Systems
"""

from . import immune
from .core import BaseClassifier
from .core import BaseClusterer
from .core import BaseOptimizer

__all__ = ["BaseClassifier", "BaseClusterer", "BaseOptimizer", "immune"]
