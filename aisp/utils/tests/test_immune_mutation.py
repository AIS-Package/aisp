"""Utility function tests for immune mutation"""

import pytest
import numpy as np
from aisp.utils.immune_mutation import generate_mutated_clones


@pytest.mark.parametrize(
    "vector, algorithm, n",
    [
        (np.array([True, False, False, True]), "binary-features", 5),
        (np.array([0.1, 0.3, 0.3, 0.4, 0.5]), "continuous-features", 5)
    ],
    ids=[
        "Generate 5 clones with binary features",
        "Generate 5 clones with continuous features"
    ]
)
def test_generate_mutated_clones(vector, algorithm, n):
    """
    Test that generate_mutated_clones returns the correct number of clones 
    for both binary and continuous feature vectors.
    """
    result = generate_mutated_clones(vector, algorithm, n)
    assert len(result) == n
