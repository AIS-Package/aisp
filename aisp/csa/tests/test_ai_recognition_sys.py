# pylint: disable=redefined-outer-name
"""Tests for classes implementing negative selection."""

import numpy as np
import pytest
from aisp.exceptions import FeatureDimensionMismatch

from aisp.csa import AIRS


@pytest.fixture
def b_airs_data():
    """Fixture that provides boolean data and a fixed seed for AIRS testing."""
    seed = 123
    np.random.seed(seed)
    X = np.random.choice([True, False], size=(500, 10))
    y = np.random.choice(["a", "b"], size=500)
    return X, y, seed


@pytest.fixture
def r_airs_data():
    """Fixture that provides real data and a fixed seed for AIRS testing."""
    seed = 123
    np.random.seed(seed)
    X = np.random.rand(500, 10)
    y = np.random.choice(["a", "b"], size=500)
    return X, y, seed


class TestAIRSBinary:
    """Test suite for the AIRS class."""

    def test_fit_and_predict(self, b_airs_data):
        """Should the fit and predict methods of the AIRS class."""
        X, y, seed = b_airs_data
        model = AIRS(algorithm="binary-features", seed=seed)
        model.fit(X, y, verbose=False)
        predictions = model.predict(X)
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_raises_feature_dimension_mismatch(self, b_airs_data):
        """Should raise FeatureDimensionMismatch when prediction input has wrong dimensions."""
        X, y, seed = b_airs_data
        model = AIRS(algorithm="binary-features", seed=seed)
        model.fit(X, y, verbose=False)
        x_invalid = np.random.choice([True, False], size=(5, 5))

        with pytest.raises(FeatureDimensionMismatch):
            model.predict(x_invalid)

    def test_score_range(self, b_airs_data):
        """Score should return a value between 0 and 1."""
        X, y, seed = b_airs_data
        model = AIRS(algorithm="binary-features", seed=seed)
        model.fit(X, y, verbose=False)
        score = model.score(X, y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestAIRSRealValue:
    """Test suite for the AIRS class."""

    def test_fit_and_predict(self, r_airs_data):
        """Should the fit and predict methods of the AIRS class."""
        X, y, seed = r_airs_data
        model = AIRS(seed=seed)
        model.fit(X, y, verbose=False)
        predictions = model.predict(X)
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_predict_raises_feature_dimension_mismatch(self, r_airs_data):
        """Should raise FeatureDimensionMismatch when prediction input has wrong dimensions."""
        X, y, seed = r_airs_data
        model = AIRS(seed=seed)
        model.fit(X, y, verbose=False)
        x_invalid = np.random.random((5, 5))
        with pytest.raises(FeatureDimensionMismatch):
            model.predict(x_invalid)

    def test_score_range(self, r_airs_data):
        """Score should return a value between 0 and 1."""
        X, y, seed = r_airs_data
        model = AIRS(seed=seed)
        model.fit(X, y, verbose=False)
        score = model.score(X, y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
