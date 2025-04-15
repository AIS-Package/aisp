"""Base Class for Negative Selection Algorithm."""
from abc import abstractmethod
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cityblock, euclidean, minkowski

from ..exceptions import FeatureDimensionMismatch
from ..utils.metrics import accuracy_score
from ..utils.sanitizers import sanitize_choice


class Base:
    """
    The base class contains functions that are used by more than one class in the package, and
    therefore are considered essential for the overall functioning of the system.

    Parameters
    ----------
    * metric (``str``): Way to calculate the distance between the detector and the sample:
        * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression: \
            √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
        * ``'minkowski'`` ➜ The calculation of the distance is given by the expression: \
            ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
        * ``'manhattan'`` ➜ The calculation of the distance is given by the expression: \
            ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) .
    * p (``float``): This parameter stores the value of ``p`` used in the Minkowski distance.\
        The default is ``2``, which represents normalized Euclidean distance. Different \
        values of p lead to different variants of the [Minkowski Distance][1].

    Notes
    ----------
    [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.minkowski_distance.html
    """

    def __init__(self, metric: str = "euclidean", p: float = 2):
        self.metric = sanitize_choice(metric, ["manhattan", "minkowski"], "euclidean")
        self.p: float = p

    def _distance(self, u: npt.NDArray, v: npt.NDArray):
        """
        Function to calculate the distance between two points by the chosen ``metric``.

        Parameters
        ----------
        * u (``npt.NDArray``): Coordinates of the first point.
        * v (``npt.NDArray``): Coordinates of the second point.

        returns
        ----------
        * Distance (``double``) between the two points.
        """
        if self.metric == "manhattan":
            return cityblock(u, v)
        if self.metric == "minkowski":
            return minkowski(u, v, self.p)

        return euclidean(u, v)

    @staticmethod
    def _check_and_raise_exceptions_fit(
            X: npt.NDArray = None,
            y: npt.NDArray = None,
            _class_: Literal["RNSA", "BNSA"] = "RNSA",
    ) -> None:
        """
        Function responsible for verifying fit function parameters and throwing exceptions if the
        verification is not successful.

        Parameters
        ----------
        * X (``npt.NDArray``) Training array, containing the samples and their
            characteristics, [``N samples`` (rows)][``N features`` (columns)].
        * y (``npt.NDArray``) Array of target classes of ``X`` with [``N samples`` (lines)].
        * _class_ (``Literal[RNSA, BNSA], optional``) Current class. Defaults to 'RNSA'.

        Raises
        ----------
        * TypeError: If X or y are not ndarrays or have incompatible shapes.
        * ValueError: If _class_ is BNSA and X contains values that are not composed only of 
            0 and 1.
        """
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)

        if not isinstance(X, np.ndarray):
            raise TypeError("X is not an ndarray or list.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y is not an ndarray or list.")

        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "X does not have the same amount of sample for the output classes in y."
            )

        if _class_ == "BNSA" and not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )

    @staticmethod
    def _check_and_raise_exceptions_predict(
        X: npt.NDArray = None,
        expected: int = 0,
        _class_: Literal["RNSA", "BNSA"] = "RNSA",
    ) -> None:
        """
        Function responsible for verifying predict function parameters and throwing exceptions if
        the verification is not successful.

        Parameters
        ----------
        * X (``npt.NDArray``)
            Input array for prediction, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].
        * expected (``int``)
            Expected number of features per sample (columns in X).
        * _class_ (``Literal[RNSA, BNSA], optional``)
            Current class. Defaults to 'RNSA'.

        Raises
        ----------
        * TypeError
            If X is not an ndarray or list.
        * FeatureDimensionMismatch
            If the number of features in X does not match the expected number.
        * ValueError
            If _class_ is BNSA and X contains values that are not composed only of 0 and 1.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError("X is not an ndarray or list")
        if expected != len(X[0]):
            raise FeatureDimensionMismatch(
                expected,
                len(X[0]),
                "X"
            )

        if _class_ != "BNSA":
            return

        # Checks if matrix X contains only binary samples. Otherwise, raises an exception.
        if not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )

    def score(self, X: npt.NDArray, y: list) -> float:
        """
        Score function calculates forecast accuracy.

        Details
        ----------
        This function performs the prediction of X and checks how many elements are equal 
        between vector y and y_predicted. This function was added for compatibility with some
        scikit-learn functions.

        Parameters
        ----------
        * X (``np.ndarray``):
            Feature set with shape (n_samples, n_features).
        * y (``np.ndarray``):
            True values with shape (n_samples,).

        Returns
        ----------
        * accuracy (``float``): The accuracy of the model.
        """
        if len(y) == 0:
            return 0
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    @abstractmethod
    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        Function to train the model using the input data ``X`` and corresponding labels ``y``.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        * X (``npt.NDArray``): Input data used for training the model, previously normalized to the
            range [0, 1].
        * y (``npt.NDArray``): Corresponding labels or target values for the input data.
        * verbose (``bool``, optional): Flag to enable or disable detailed output during training.
            Default is ``True``.

        Returns
        ----------
        * self: Returns the instance of the class that implements this method.
        """

    @abstractmethod
    def predict(self, X) -> Optional[npt.NDArray]:
        """
        Function to generate predictions based on the input data ``X``.

        This abstract method is implemented by the class that inherits it.

        Parameters
        ----------
        * X (``npt.NDArray``): Input data for which predictions will be generated.

        Returns
        ----------
        * Predictions (``Optional[npt.NDArray]``): Predicted values for each input sample, or
            ``None`` if the prediction fails.
        """
