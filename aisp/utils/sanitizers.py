"""Utility functions for validation and treatment of parameters."""

from typing import TypeVar, Iterable, Callable, Any, Optional, Dict

import numpy as np
import numpy.typing as npt

T = TypeVar('T')


def sanitize_choice(value: T, valid_choices: Iterable[T], default: T) -> T:
    """
    Value if present in the set of valid choices; otherwise, the default value.

    Parameters
    ----------
    value : T
        The value to be checked.
    valid_choices : Iterable[T]
        A collection of valid choices.
    default : T
        The default value to be returned if 'value' is not in 'valid_choices'.

    Returns
    -------
    value : T
        The original value if valid, or the default value if not.
    """
    return value if value in valid_choices else default


def sanitize_param(value: T, default: T, condition: Callable[[T], bool]) -> T:
    """
    Value if it satisfies the specified condition; otherwise, the default value.

    Parameters
    ----------
    value : T
        The value to be checked.
    default : T
        The default value to be returned if the condition is not satisfied.
    condition : Callable[[T], bool]
        A function that takes a value and returns a boolean, determining if the value is valid.

    Returns
    -------
    value : T
        The original value if the condition is satisfied, or the default value if not.
    """
    return value if condition(value) else default


def sanitize_seed(seed: Any) -> Optional[int]:
    """
    Seed if it is a non-negative integer; otherwise, None.

    Parameters
    ----------
    seed : Any
        The seed value to be validated.

    Returns
    -------
    seed : Optional[int]
        The original seed if it is a non-negative integer, or None if it is invalid.
    """
    return seed if isinstance(seed, int) and seed >= 0 else None


def sanitize_bounds(bounds: Any, problem_size: int) -> Dict[str, npt.NDArray[np.float64]]:
    """Validate and normalize feature bounds.

    Parameters
    ----------
    bounds : Any
        The input bounds, which must be either None or a dictionary with 'low'
        and 'high' keys.
    problem_size : int
        The expected length for the normalized bounds lists, corresponding to
        the number of features in the problem.

    Returns
    -------
    Dict[str, list]
        Dictionary {'low': [low_1, ..., low_N], 'high': [high_1, ..., high_N]}.

    Raises
    ------
    TypeError
        If `bounds` is not None and not a dict with 'low'/'high', or if items are non-numeric.
    ValueError
        If provided iterables have the wrong length.
    """
    if bounds is None or not isinstance(bounds, dict) or set(bounds.keys()) != {'low', 'high'}:
        raise ValueError("bounds expects a dict with keys 'low' and 'high'")
    result = {}

    for key in ['low', 'high']:
        value = bounds[key]
        if isinstance(value, (float, int)):
            result[key] = np.array([value] * problem_size).astype(dtype=np.float64)
        else:
            if not isinstance(value, (list, np.ndarray)):
                raise TypeError(
                    f"{key} must be a list or numpy array, got {type(value).__name__}"
                )
            if not all(isinstance(i, (float, int)) for i in value):
                raise TypeError(f"All elements of {key} must be numeric")

            value = np.array(value).astype(dtype=np.float64)
            if len(value) != problem_size:
                raise ValueError(
                    f"The size of {key} must be equal to the size of the problem ({problem_size})"
                )
            result[key] = value
    return result
