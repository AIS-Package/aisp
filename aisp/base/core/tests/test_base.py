# pylint: disable=attribute-defined-outside-init
"""Unit tests for the Base class."""

from aisp.base.core._base import Base


class TestBase:
    """Unit tests for the Base class."""

    def setup_method(self):
        """Set up a Base instance with example attributes before each test."""
        self.obj = Base()
        self.obj.alpha = 1
        self.obj.beta = 2

    def test_set_and_get_params_basic(self):
        """Test setting parameters using set_params and retrieving them with get_params."""
        params_dict = {"alpha": 10, "beta": "test"}
        self.obj.set_params(**params_dict)

        params = self.obj.get_params()
        assert params == params_dict

    def test_get_params_excludes_private(self):
        """Test that get_params excludes attributes starting with an underscore."""
        self.obj._private = 123  # pylint: disable=protected-access
        self.obj.public = "ok"

        params = self.obj.get_params()
        assert "_private" not in params
        assert "public" in params
        assert params["public"] == "ok"

    def test_set_params_updates_existing(self):
        """Test that set_params updates existing attributes correctly."""
        self.obj.alpha = 5
        self.obj.set_params(alpha=42)

        assert self.obj.alpha == 42

    def test_set_params_not_hasattr(self):
        """Test that set_params ignores parameters not corresponding to existing attributes."""
        self.obj.set_params(not_params=42)

        params = self.obj.get_params()
        assert "not_params" not in params
