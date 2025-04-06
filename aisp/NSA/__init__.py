"""Module (NSA) Negative Selection Algorithm

NSAs simulate the maturation process of T-cells in the immune system, where these \
cells learn to distinguish between self and non-self. Only T lymphocytes capable \
of recognizing non-self elements are preserved, helping the system detect invaders.

----

Os NSAs simulam o processo de maturação das células-T no sistema imunológico, onde \
essas células aprendem a distinguir entre o próprio e não-próprio.
Apenas as linfócitos-T capazes de reconhecer elementos não-próprios são preservadas,\
ajudando o sistema a detectar invasores.

"""
from ._negative_selection import BNSA, RNSA

__author__ = "João Paulo da Silva Barros"
__all__ = ["RNSA", "BNSA"]
__version__ = "0.1.34"
