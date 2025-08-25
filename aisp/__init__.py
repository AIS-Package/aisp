"""AISP - Artificial Immune Systems Package.

AISP is a Python package of immunoinspired techniques that apply metaphors from the vertebrate
immune system to pattern recognition and optimization tasks.

The package is organized into specialized modules, each dedicated to a family of Artificial
Immune Systems algorithms:
- csa: Clonal Selection Algorithms
- nsa: Negative Selection Algorithms
- ina: Immune Network Algorithms

For detailed documentation and examples, visit:
https://ais-package.github.io/docs/intro
"""

from . import csa
from . import nsa
from . import ina

__author__ = "AISP Development Team"
__version__ = "0.3.1"
__all__ = [
    'csa',
    'nsa',
    'ina'
]
