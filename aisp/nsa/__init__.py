"""Module (NSA) Negative Selection Algorithm.

NSAs simulate the maturation process of T-cells in the immune system, where these cells learn to 
distinguish between self and non-self. Only T-cells capable of recognizing non-self elements are 
preserved.
"""

from ._binary_negative_selection import BNSA
from ._negative_selection import RNSA

__author__ = "João Paulo da Silva Barros"
__all__ = ["RNSA", "BNSA"]
