<div align = center>

<img alt="Artificial Immune Systems Package" src='https://ais-package.github.io/assets/images/logo-7b415c6841a3ed8a760eff38ecd996b8.svg'/>

# Artificial Immune Systems Package

A Python package for Artificial Immune Systems algorithms

</div>

---

## Language

* [Português](https://github.com/AIS-Package/aisp/blob/main/docs/pt-br/README.md)

## Documentation

* [Official Docs](https://ais-package.github.io/docs/intro)
* [Github Wiki](https://github.com/AIS-Package/aisp/wiki)

---

## Introduction

**AISP** is a python package that implements artificial immune systems techniques, distributed under the GNU Lesser
General Public License v3.0 (LGPLv3).

The package started in **2022** as a research package at the Federal Institute of Northern Minas Gerais - Salinas
campus (**IFNMG - Salinas**).

Artificial Immune Systems (AIS) are inspired by the vertebrate immune system, creating metaphors that apply the
ability to detect and catalog pathogens, among other features of this system.

### What can you do with AISP?

AISP provides implementations of bio-inspired algorithms for:

- **Anomaly detection:** Identify abnormal patterns in data.
- **Classification:** Classify data with multiple classes.
- **Optimization:** Find optimal solutions for objective functions.
- **Clustering:** Group data without supervision.

---

## Implemented Algorithms

### Negative Selection (`aisp.nsa`)

- **BNSA** - Binary Negative Selection Algorithm  
- **RNSA** - Real-Valued Negative Selection Algorithm  

### Clonal Selection (`aisp.csa`)

- **AIRS** - Artificial Immune Recognition System  
- **CLONALG** - Clonal Selection Algorithm  

### Immune Network Theory (`aisp.ina`)

- **AiNet** - Artificial Immune Network for clustering and data compression

### Module in Development

#### Danger Theory (`aisp.dta`)

- **DCA** - Dendritic Cell Algorithm *(planned)*

## API overview

All algorithms follow a simple and consistent interface:

- `fit(X, y, verbose: bool = True)`: trains the model for classification tasks.
- `fit(X, verbose: bool = True)`: trains the model for clustering tasks.
- `predict(X)`: makes predictions based on new data.
- `optimize(max_iters: int =..., n_iter_no_change: int =..., verbose: bool = True)`: run the optimization algorithms

---

## Installation

The module requires installation of [python 3.10](https://www.python.org/downloads/) or higher.

### Dependencies

<div align = center>

| Packages |  Version  |
|:--------:|:---------:|
|  numpy   | ≥ 1.22.4  |
|  scipy   |  ≥ 1.8.1  |
|   tqdm   | ≥ 4.64.1  |
|  numba   | ≥ 0.59.0  |

</div>

### User installation

The simplest way to install AISP is using ``pip``:

```Bash
pip install aisp
```

---

## Quick Start

Below are minimal examples demonstrating how to use AISP for different tasks.

### Classification with RNSA

```python
import numpy as np
from aisp.nsa import RNSA

# Generating training data
np.random.seed(1)
class_a = np.random.uniform(high=0.5, size=(50, 2))
class_b = np.random.uniform(low=0.51, size=(50, 2))
x_train = np.vstack((class_a, class_b))
y_train = ['a'] * 50 + ['b'] * 50

# Training the model
model = RNSA(N=150, r=0.3, seed=1)
model.fit(x_train, y_train, verbose=False)

# Predict
x_test = [
    [0.15, 0.45],  # Expected: 'a'
    [0.85, 0.65],  # Expected: 'b'
]

y_pred = model.predict(x_test)
print(y_pred)
```

### Clustering with AiNet

```python
import numpy as np
from aisp.ina import AiNet

np.random.seed(1)
# Generating training data
a = np.random.uniform(high=0.4, size=(50, 2))
b = np.random.uniform(low=0.6, size=(50, 2))
x_train = np.vstack((a, b))

# Training the model
model = AiNet(
    N=150,
    mst_inconsistency_factor=1,
    seed=1,
    affinity_threshold=0.85,
    suppression_threshold=0.7
)

model.fit(x_train, verbose=False)

# Predict cluster labels
x_test = [
    [0.15, 0.45],
    [0.85, 0.65],
]

y_pred = model.predict(x_test)
print(y_pred)
```

### Optimization with CLONALG

```python
import numpy as np
from aisp.csa import Clonalg

# Define search space
bounds = {'low': -5.12, 'high': 5.12}

# Objective function (Rastrigin)
def rastrigin(x):
    x = np.clip(x, bounds['low'], bounds['high'])
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

# Initialize optimizer
model = Clonalg(problem_size=2, rate_hypermutation=0.5, bounds=bounds, seed=1)
model.register('affinity_function', rastrigin)

# Run optimization
population = model.optimize(100, 50, False)

print(model.best_solution, model.best_cost) # Best solution
```

---

## Examples

Explore the example notebooks available in the [AIS-Package/aisp repository](https://github.com/AIS-Package/aisp/tree/main/examples).
These notebooks demonstrate how to utilize the package's functionalities in various scenarios, including applications of the RNSA,
BNSA and AIRS algorithms on datasets such as Iris, Geyser, and Mushrooms.

You can run the notebooks directly in your browser without any local installation using Binder:

[![Launch on Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AIS-Package/aisp/HEAD?labpath=%2Fexamples)

💡 **Tip**: Binder may take a few minutes to load the environment, especially on the first launch.
