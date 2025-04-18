"""Unit tests for the detector validity checking functions implementation."""
import pytest
import numpy as np

from aisp.nsa._ns_core import bnsa_class_prediction, check_detector_bnsa_validity

@pytest.mark.parametrize(
    "x_class, vector_x, aff_thresh, expected_result",
    [
        (np.array([[1, 1, 0], [0, 0, 1]]), np.array([1, 0, 0]), 0.33, True),
        (np.array([[1, 1, 0], [0, 0, 1]]), np.array([1, 1, 0]), 0.33, False),
        (np.array([[1, 0, 1], [0, 1, 0]]), np.array([0, 1, 1]), 0.9, False),
        (np.array([[1, 1, 0], [0, 0, 1]]), np.array([1, 0]), 0.33, False),
    ],
    ids=[
        'The Hamming distance is greater than the affinity threshold.',
        'The Hamming distance is less than or equal to the affinity threshold.',
        'Affinity threshold is too high.',
        "The detector vector has a size mismatch with the samples."
    ]
)
def test_check_detector_bnsa_validity(
    x_class: np.ndarray,
    vector_x: np.ndarray,
    aff_thresh: float,
     expected_result: bool
):
    """
    The check_detector_bnsa_validity function tests whether the normalized Hamming 
    distance falls within the affinity limit.
    """
    result = check_detector_bnsa_validity(x_class, vector_x, aff_thresh)
    assert result == expected_result

@pytest.mark.parametrize(
    "features, class_detectors, aff_thresh, expected",
    [
        (
            np.array([1, 0, 0]),
            np.array([
                [[1, 1, 1], [1, 1, 1]],
                [[0, 0, 1], [0, 1, 0]]
            ]),
            0.25,
            0
        ),
        (
            np.array([0, 1, 1]),
            np.array([
                [[0, 0, 1], [1, 0, 1]],
                [[0, 1, 0], [0, 1, 1]]
            ]),
            0.6,
            -1
        )
    ],
    ids=[
        "Sample suitable only for class 0",
        "Sample rejected by all detectors (non-suitable)"
    ]
)
def test_bnsa_class_prediction(features, class_detectors, aff_thresh, expected):
    """
    Tests the function that predicts the classes for binary samples by the trained detectors.
    """
    result = bnsa_class_prediction(features, class_detectors, aff_thresh)
    assert result == expected
