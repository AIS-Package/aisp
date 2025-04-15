"""Tests for classes implementing negative selection."""

import numpy as np
from aisp.nsa import BNSA, RNSA


class TestBNSA:
    """Test suite for the BNSA class."""
    def __init__(self):
        self.seed = 123
        np.random.seed(self.seed)
        self.X = np.random.choice([True, False], size=(100, 10))
        self.y = np.random.choice(["a", "b"], size=100)

    def test_fit_predict(self):
        """Test the fit and predict methods of the BNSA class."""
        model = BNSA(N=50, aff_thresh=0.2, seed=self.seed)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert predictions is not None
        assert len(predictions) == len(self.y)

    def test_score(self):
        """Test the score method of the BNSA class."""
        model = BNSA(N=50, aff_thresh=0.2, seed=self.seed)
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)
        assert 0.0 <= score <= 1.0


class TestRNSA:
    """Test suite for the RNSA class."""

    def __init__(self):
        self.seed = 123
        np.random.seed(self.seed)
        self.X = np.random.rand(100, 10)
        self.y = np.random.choice(["a", "b"], size=100)

    def test_fit_predict(self):
        """Test the fit and predict methods of the RNSA class."""
        model = RNSA(N=50, r=0.1, seed=self.seed)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        assert predictions is not None
        assert len(predictions) == len(self.y)

    def test_score(self):
        """Test the score method of the RNSA class."""
        model = RNSA(N=50, r=0.1, seed=self.seed)
        model.fit(self.X, self.y)
        score = model.score(self.X, self.y)
        assert 0.0 <= score <= 1.0
