"""Custom warnings and errors."""

from typing import Optional


class MaxDiscardsReachedError(Exception):
    """Exception thrown when the maximum number of detector discards is reached."""

    def __init__(self, _class_, message=None):
        if message is None:
            message = (
                "An error has been identified:\n"
                f"the maximum number of discards of detectors for the {_class_} class "
                "has been reached.\nIt is recommended to check the defined radius and "
                "consider reducing its value."
            )

        super().__init__(message)


class FeatureDimensionMismatch(Exception):
    """
    Exception raised when the number of input features does not match the expected number.

    This exception is triggered during prediction if the input features' dimension is incorrect.
    """

    def __init__(
        self, expected: int, received: int, variable_name: Optional[str] = None
    ):
        parts = []
        if variable_name:
            parts.append(f"In variable '{variable_name}'")

        parts.append("feature dimension mismatch")

        message = (
            f"{' '.join(parts)}: expected {expected} features, but received {received}. "
            "Please ensure the input data has the correct number of features "
            "and matches the expected shape for the model."
        )
        super().__init__(message)


class UnsupportedTypeError(Exception):
    """
    Exception raised when the input vector type is not supported.

    This exception is thrown when the vector data type does not match any of the supported.
    """

    def __init__(self, message=None):
        if message is None:
            message = (
                "Type is not supported. Provide a binary, normalized, or bounded "
                "continuous vector."
            )
        super().__init__(message)


class ModelNotFittedError(Exception):
    """
    Exception raised when a method is called before the model has been fit.

    This exception is thrown when the  model instance is being used without first training
    it via the `fit` method.
    """

    def __init__(self, object_name: str, message: Optional[str] = None):
        if message is None:
            message = (
                f"The model {object_name} must be fitted before use. Train the model by calling "
                "the fit method before using it."
            )
        super().__init__(message)
