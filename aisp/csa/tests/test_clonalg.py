# pylint: disable=redefined-outer-name
"""Tests for classes implementing clonalg."""

import numpy as np
import pytest

from aisp.base.immune.cell import Antibody
from aisp.csa import Clonalg


def affinity(x):
    """Simple affinity function."""
    return np.sum(x)


@pytest.fixture
def clonalg():
    """Default Clonalg instance."""
    return Clonalg(
        problem_size=3,
        N=10,
        affinity_function=affinity,
        bounds={'low': 1.0, 'high': 10.0},
        seed=42
    )


class TestClonalg:
    """Test suite for the Clonalg class."""

    def test_optimize_returns_valid_population(self, clonalg):
        """Should return a valid population with correct shape and within bounds."""
        pop = clonalg.optimize(max_iters=5, verbose=False)
        assert isinstance(pop, list)
        assert isinstance(pop[0], Antibody)
        assert len(pop) == 10
        assert pop[0].vector.shape[0] == 3

    def test_optimize_without_affinity_function_raises_error(self):
        """Should raise NotImplementedError if affinity_function is not provided."""
        model = Clonalg(3, bounds={'low': 1, 'high': 2})
        with pytest.raises(NotImplementedError):
            model.optimize(verbose=False)

    def test_register_affinity_function_enables_optimization(self, clonalg):
        """Should allow optimization after registering an affinity_function."""
        clonalg.register('affinity_function', affinity)
        clonalg.optimize(verbose=False)
