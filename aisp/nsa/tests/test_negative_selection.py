# pylint: disable=redefined-outer-name
"""Tests for classes implementing negative selection."""

import numpy as np
import pytest

from aisp.exceptions import FeatureDimensionMismatch, MaxDiscardsReachedError
from aisp.nsa import BNSA, RNSA


@pytest.fixture
def bnsa_data():
    """Fixture that provides boolean data and a fixed seed for BNSA testing."""
    seed = 123
    np.random.seed(seed)
    X = np.random.choice([True, False], size=(100, 10))
    y = np.random.choice(["a", "b"], size=100)
    return X, y, seed


@pytest.fixture
def rnsa_data():
    """Fixture that provides real data and a fixed seed for RNSA testing."""
    seed = 123
    np.random.seed(seed)
    X = np.random.rand(100, 10)
    y = np.random.choice(["a", "b"], size=100)
    return X, y, seed


class TestBNSA:
    """Test suite for the BNSA class."""

    def test_fit_and_predict(self, bnsa_data):
        """Should the fit and predict methods of the BNSA class."""
        X, y, seed = bnsa_data
        model = BNSA(N=50, aff_thresh=0.2, seed=seed)
        model.fit(X, y, verbose=False)
        predictions = model.predict(X)
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_fit_raises_max_discards_error(self, bnsa_data):
        """Should raise MaxDiscardsReachedError when max_discards is exceeded during fit."""
        X, y, seed = bnsa_data
        model = BNSA(N=1000, aff_thresh=0.2, seed=seed, max_discards=2)
        expected_message = (
            "An error has been identified:\n"
            "the maximum number of discards of detectors for the a class "
            "has been reached.\nIt is recommended to check the defined radius and "
            "consider reducing its value."
        )
        with pytest.raises(MaxDiscardsReachedError, match=expected_message):
            model.fit(X, y, verbose=False)

    def test_predict_raises_feature_dimension_mismatch(self, bnsa_data):
        """Should raise FeatureDimensionMismatch when prediction input has wrong dimensions."""
        X, y, seed = bnsa_data
        model = BNSA(N=1000, aff_thresh=0.2, seed=seed)
        model.fit(X, y, verbose=False)
        x_invalid = np.random.choice([True, False], size=(5, 5))

        with pytest.raises(FeatureDimensionMismatch):
            model.predict(x_invalid)

    def test_score_range(self, bnsa_data):
        """Score should return a value between 0 and 1."""
        X, y, seed = bnsa_data
        model = BNSA(N=50, aff_thresh=0.2, seed=seed)
        model.fit(X, y, verbose=False)
        score = model.score(X, y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestRNSA:
    """Test suite for the RNSA class."""

    def test_fit_and_predict(self, rnsa_data):
        """Should the fit and predict methods of the RNSA class."""
        X, y, seed = rnsa_data
        model = RNSA(N=50, r=0.1, seed=seed)
        model.fit(X, y, verbose=False)
        predictions = model.predict(X)
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)

    def test_fit_raises_max_discards_error(self, rnsa_data):
        """Should raise MaxDiscardsReachedError when max_discards is exceeded during fit."""
        X, y, seed = rnsa_data
        model = RNSA(N=1000, r=0.9, r_s=0.5, seed=seed, max_discards=2)
        expected_message = (
            "An error has been identified:\n"
            "the maximum number of discards of detectors for the a class "
            "has been reached.\nIt is recommended to check the defined radius and "
            "consider reducing its value."
        )
        with pytest.raises(MaxDiscardsReachedError, match=expected_message):
            model.fit(X, y, verbose=False)

    def test_predict_raises_feature_dimension_mismatch(self, rnsa_data):
        """Should raise FeatureDimensionMismatch when prediction input has wrong dimensions."""
        X, y, seed = rnsa_data
        model = RNSA(N=1000, aff_thresh=0.2, seed=seed)
        model.fit(X, y, verbose=False)
        x_invalid = np.random.rand(5, 5)
        with pytest.raises(FeatureDimensionMismatch):
            model.predict(x_invalid)

    def test_score_range(self, rnsa_data):
        """Score should return a value between 0 and 1."""
        X, y, seed = rnsa_data
        model = RNSA(N=50, r=0.1, seed=seed)
        model.fit(X, y, verbose=False)
        score = model.score(X, y)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
