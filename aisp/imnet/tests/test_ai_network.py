# pylint: disable=redefined-outer-name
"""Tests for classes implementing artificial immune network."""

import numpy as np
import pytest
from aisp.exceptions import FeatureDimensionMismatch

from aisp.imnet import AiNet


@pytest.fixture
def b_ai_net_data():
    """Fixture that provides boolean data and a fixed seed for AiNet testing."""
    seed = 123
    np.random.seed(seed)
    X = np.random.choice([True, False], size=(500, 2))
    return X, seed


@pytest.fixture
def r_ai_net_data():
    """Fixture that provides real data and a fixed seed for AiNet testing."""
    seed = 123
    np.random.seed(seed)
    X = np.random.rand(500, 2)
    return X, seed


class TestAiNetBinary:
    """Test suite for the AiNet class."""

    def test_fit_and_predict(self, b_ai_net_data):
        """Should the fit and predict methods of the AiNet class."""
        X, seed = b_ai_net_data
        model = AiNet(seed=seed)
        predictions = model.fit_predict(X, verbose=False)
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == X.shape[0]

    def test_predict_raises_feature_dimension_mismatch(self, b_ai_net_data):
        """Should raise FeatureDimensionMismatch when prediction input has wrong dimensions."""
        X, seed = b_ai_net_data
        model = AiNet(seed=seed)
        model.fit(X, verbose=False)
        x_invalid = np.random.choice([True, False], size=(5, 5))

        with pytest.raises(FeatureDimensionMismatch):
            model.predict(x_invalid)


class TestAiNetRealValue:
    """Test suite for the AiNet class."""

    def test_fit_and_predict(self, r_ai_net_data):
        """Should the fit and predict methods of the AiNet class."""
        X, seed = r_ai_net_data
        model = AiNet(seed=seed)
        predictions = model.fit_predict(X, verbose=False)
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == X.shape[0]

    def test_predict_raises_feature_dimension_mismatch(self, r_ai_net_data):
        """Should raise FeatureDimensionMismatch when prediction input has wrong dimensions."""
        X, seed = r_ai_net_data
        model = AiNet(seed=seed)
        model.fit(X, verbose=False)
        x_invalid = np.random.random((5, 5))
        with pytest.raises(FeatureDimensionMismatch):
            model.predict(x_invalid)
