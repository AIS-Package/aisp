"""Module (ina) Immune Network Algorithm.

This module implements algorithms based on Network Theory Algorithms proposed by Jerne.

Classes
-------
AiNet : Artificial Immune Network.
    An unsupervised learning algorithm for clustering, based on the theory of immune networks.
"""

from ._ai_network import AiNet
from ._opt_ai_network import OptAiNet

__all__ = ["AiNet", "OptAiNet"]
