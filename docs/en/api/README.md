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

## Core Modules (`aisp.base`)

Fundamental abstractions and base classes that define core interfaces of package.

| Class                                         | Description                                        |
|-----------------------------------------------|----------------------------------------------------|
| [`BaseClassifier`](./base/base-classifier.md) | Abstract base class for classification algorithms. |
| [`BaseClusterer`](./base/base-clusterer.md)   | Abstract base class for clustering algorithms.     |
| [`BaseOptimizer`](./base/base-optimizer.md)   | Abstract base class for optimization algorithms.   |

### Immune Domain components (`aisp.base.immune`)

Core structures and support utilities for implementations immune.

| Class                                         | Description                                                                                |
|-----------------------------------------------|--------------------------------------------------------------------------------------------|
| [`cell`](./base/immune/cell/README.md)        | Representation of immune system cells.                                                     |
| [`mutation`](./base/immune/mutation.md)       | Functions to generate mutated clones and simulate clonal expansion.                        |
| [`populations`](./base/immune/populations.md) | Provide utility functions for generating antibody populations in immunological algorithms. |

---

## Algorithms

### Negative Selection Algorithms (`aisp.nsa`)

supervised learning algorithms based on negative selection, the immune systems process of distinguishing self from not-self.

| Class                   | Description                                                                         |
|-------------------------|-------------------------------------------------------------------------------------|
| [`RNSA`](./nsa/rnsa.md) | A supervised learning algorithm for classification that uses real-valued detectors. |
| [`BNSA`](./nsa/bnsa.md) | A supervised learning algorithm for classification that uses binary detectors.      |

### Clonal Selection Algorithms (`aisp.csa`)

Algorithms inspired by the process of antibody proliferation to detecting an antigen.

| Class                         | Description                                                                                                                                                                          |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`AIRS`](./csa/airs.md)       | A supervised learning algorithm for classification tasks based on the clonal selection principle.                                                                                    |
| [`Clonalg`](./csa/clonalg.md) | Implementation of the clonal selection algorithm for optimization, adapted for both minimization and maximization of cost functions in binary, continuous, and permutation problems. |

### Immune Network Algorithms  (`aisp.ina`)

Algorithms based on Network Theory Algorithms proposed by Jerne.

| Class                  | Description                                                                                |
|------------------------|--------------------------------------------------------------------------------------------|
| [`AiNet`](./ai-net.md) | An unsupervised learning algorithm for clustering, based on the theory of immune networks. |

## Utils (`aisp.utils`)