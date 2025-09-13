"""Module (CSA) Clonal Selection Algorithm.

CSAs are inspired by the process of antibody proliferation upon detecting an antigen, during which
the generated antibodies undergo mutations in an attempt to enhance pathogen recognition.

Classes
-------
AIRS : Artificial Immune Recognition System.
    A supervised learning algorithm for classification tasks based on the clonal
    selection principle.
Clonalg : Clonal Selection Algorithm.
    Classical implementation of the clonal selection algorithm, adapted for minimizing cost
    functions in binary, continuous, and interval-valued problem
"""
from ._ai_recognition_sys import AIRS
from ._clonalg import Clonalg

__author__ = 'João Paulo da Silva Barros'
__all__ = ['AIRS', 'Clonalg']
