"""
Defines type aliases used throughout the project to improve readability.

FeatureType:
    Type of input features:
    - "binary-features": values like 0 or 1.
    - "continuous-features": numeric continuous values.
    - "ranged-features": values defined by intervals.

FeatureTypeAll:
    Same as ``FeatureType``, plus:
    - "permutation-features": values represented as permutation.

MetricType:
    Distance metric used in calculations:
    - "manhattan": the Manhattan distance between two points
    - "minkowski": the Minkowski distance between two points.
    - "euclidean": the Euclidean distance between two points.
"""

from typing import Literal, TypeAlias, Union

FeatureType: TypeAlias = Literal[
    "binary-features", "continuous-features", "ranged-features"
]

FeatureTypeAll: TypeAlias = Union[FeatureType, Literal["permutation-features"]]

MetricType: TypeAlias = Literal["manhattan", "minkowski", "euclidean"]
