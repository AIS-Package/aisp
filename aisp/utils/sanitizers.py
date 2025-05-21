"""Utility functions for validation and treatment of parameters."""
from typing import TypeVar, Iterable, Callable, Any, Optional

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
