"""AISP - Artificial Immune Systems Package.

AISP is a Python package of immunoinspired techniques that apply metaphors from the vertebrate
immune system to pattern recognition and optimization tasks.

The package is organized into specialized modules, each dedicated to a family of Artificial
Immune Systems algorithms:

Modules
-------
csa : Clonal Selection Algorithms.
    Inspired by the processes of antibody proliferation and mutation.
    - AIRS: Artificial Immune Recognition System for classification.
    - Clonalg: Clonal Selection Algorithm for optimization.

nsa : Negative Selection Algorithms
    Simulates T cell maturation and is capable of detecting non-self cells.
    - RNSA: Real-value Negative Selection Algorithm for classification.
    - BNSA: Binary Negative Selection Algorithm for classification.

ina : Immune Network Algorithms
    Based on immune network theory.
    - AiNet: Artificial Immune Network for clustering.

For detailed documentation and examples, visit:
https://ais-package.github.io/docs/intro
"""

from . import csa
from . import ina
from . import nsa

__author__ = "AISP Development Team"
__version__ = "0.4.0"
__all__ = ["csa", "nsa", "ina"]
