---
id: api
sidebar_label: api
keywords:
    - api
    - aisp
    - Artificial Immune System Package
    - classification
    - optimization
    - clustering
---

# AISP - API

Welcome to the **AISP** (Artificial Immune System Package) api. This documentation demonstrates the package public API.

## Core Modules ([`aisp.base`](./base/README.md))

Fundamental abstractions and base classes that define core interfaces of package.

| Class                                         | Description                                        |
|-----------------------------------------------|----------------------------------------------------|
| [`BaseClassifier`](./base/base-classifier.md) | Abstract base class for classification algorithms. |
| [`BaseClusterer`](./base/base-clusterer.md)   | Abstract base class for clustering algorithms.     |
| [`BaseOptimizer`](./base/base-optimizer.md)   | Abstract base class for optimization algorithms.   |

### Immune Domain components ([`aisp.base.immune`](./base/immune/README.md))

Core structures and support utilities for implementations immune.

| Module                                        | Description                                                                                |
|-----------------------------------------------|--------------------------------------------------------------------------------------------|
| [`cell`](./base/immune/cell/README.md)        | Representation of immune system cells.                                                     |
| [`mutation`](./base/immune/mutation.md)       | Functions to generate mutated clones and simulate clonal expansion.                        |
| [`populations`](./base/immune/populations.md) | Provide utility functions for generating antibody populations in immunological algorithms. |

---

## Algorithms

### Negative Selection Algorithms ([`aisp.nsa`](./nsa/README.md))

supervised learning algorithms based on negative selection, the immune systems process of distinguishing self from not-self.

| Class                   | Description                                                                         |
|-------------------------|-------------------------------------------------------------------------------------|
| [`RNSA`](./nsa/rnsa.md) | A supervised learning algorithm for classification that uses real-valued detectors. |
| [`BNSA`](./nsa/bnsa.md) | A supervised learning algorithm for classification that uses binary detectors.      |

### Clonal Selection Algorithms ([`aisp.csa`](./csa/README.md))

Algorithms inspired by the process of antibody proliferation to detecting an antigen.

| Class                         | Description                                                                                                                                                                          |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`AIRS`](./csa/airs.md)       | A supervised learning algorithm for classification tasks based on the clonal selection principle.                                                                                    |
| [`Clonalg`](./csa/clonalg.md) | Implementation of the clonal selection algorithm for optimization, adapted for both minimization and maximization of cost functions in binary, continuous, and permutation problems. |

### Immune Network Algorithms  ([`aisp.ina`](./ina/README.md))

Algorithms based on Network Theory Algorithms proposed by Jerne.

| Class                  | Description                                                                                |
|------------------------|--------------------------------------------------------------------------------------------|
| [`AiNet`](./ai-net.md) | An unsupervised learning algorithm for clustering, based on the theory of immune networks. |

## Utils ([`aisp.utils`](./utils/README.md))

Utility functions and helpers for development.

| Module                                 | Description                                                              |
|----------------------------------------|--------------------------------------------------------------------------|
| [`display`](./utils/display/README.md) | Utility functions for displaying algorithm information.                  |
| [`distance`](./utils/distance.md)      | Utility functions for distance between arrays with numba decorators.     |
| [`metrics`](./utils/metrics.md)        | Utility functions for measuring accuracy and performance.                |
| [`multiclass`](./utils/multiclass.md)  | Utility functions for handling classes with multiple categories.         |
| [`sanitizers`](./utils/sanitizers.md)  | Utility functions for validation and treatment of parameters.            |
| [`types`](./utils/types.md)            | Defines type aliases used throughout the project to improve readability. |
| [`validation`](./utils/validation.md)  | Contains functions responsible for validating data types.                |