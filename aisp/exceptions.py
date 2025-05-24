"""Custom warnings and errors."""


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
        self,
        expected: int,
        received: int,
        variable_name: str = None
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
