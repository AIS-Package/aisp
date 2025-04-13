"""nsa: Module (NSA) Negative Selection Algorithm

NSAs simulate the maturation process of T-cells in the immune system, where these cells learn to 
distinguish between self and non-self. Only T-cells capable of recognizing non-self elements are 
preserved.
"""
from ._negative_selection import BNSA, RNSA

__author__ = "João Paulo da Silva Barros"
__all__ = ["RNSA", "BNSA"]
__version__ = "0.1.34"
