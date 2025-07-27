"""
Defines type aliases used throughout the project to improve readability.

Type Aliases
------------
FeatureType : Literal["binary-features", "continuous-features", "ranged-features"]
    Specifies the type of features in the input data. Can be one of:
    - "binary-features": Features with binary values (e.g., 0 or 1).
    - "continuous-features": Features with continuous numeric values.
    - "ranged-features": Features represented by ranges or intervals.

MetricType : Literal["manhattan", "minkowski", "euclidean"]
    Specifies the distance metric to use for calculations. Possible values:
    - "manhattan": The calculation of the distance is given by the expression:
            √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
    - "minkowski": The calculation of the distance is given by the expression:
            ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
    - "euclidean": The calculation of the distance is given by the expression:
            ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|).
"""


from typing import Literal, TypeAlias


FeatureType: TypeAlias = Literal[
    "binary-features",
    "continuous-features",
    "ranged-features"
]
MetricType: TypeAlias = Literal["manhattan", "minkowski", "euclidean"]
