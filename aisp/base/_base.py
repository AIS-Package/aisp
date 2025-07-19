"""Base class for objects with parameter extraction support."""


class Base:
    def get_params(self, deep: bool = True) -> dict:  # pylint: disable=W0613
        """
        Return a dictionary with the object's main parameters.

        This method is required to ensure compatibility with scikit-learn functions.

        Returns
        -------
        dict
            Dictionary containing the object's attributes that do not start with "_".
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }
