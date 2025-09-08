"""Module (CSA) Clonal Selection Algorithm.

CSAs are inspired by the process of antibody proliferation upon detecting an antigen, during which
the generated antibodies undergo mutations in an attempt to enhance pathogen recognition.
"""
from ._ai_recognition_sys import AIRS
from ._clonalg import Clonalg

__author__ = 'João Paulo da Silva Barros'
__all__ = ['AIRS', 'Clonalg']
