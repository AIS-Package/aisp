"""Module (NSA) Negative Selection Algorithm.

NSAs simulate the maturation process of T-cells in the immune system, where these cells learn to 
distinguish between self and non-self. Only T-cells capable of recognizing non-self elements are 
preserved.

Classes
-------
RNSA : Real-valued Negative Selection Algorithm.
    A supervised learning algorithm for classification that uses real-valued detectors.
BNSA : Binary Negative Selection Algorithm.
    A supervised learning algorithm for classification that uses binary detectors.
"""

from ._binary_negative_selection import BNSA
from ._negative_selection import RNSA

__all__ = ["RNSA", "BNSA"]
