---
id: faq
sidebar_position: 6
sidebar_label: FAQ
keywords:
    - FAQ
    - Frequently Asked Questions
    - Questions
    - Help
---

# Frequently Asked Questions

Quick solutions for possible questions about aisp.

## General usage

### Which algorithm should I choose?

It depends on the type of problem:

- **Anomaly detection**: Use `RNSA` or `BNSA`.
  - RNSA for problems with continuous data.
  - BNSA for problems with binary data.
- **Classification**: Use `AIRS`, `RNSA`, or `BNSA`.
  - `RNSA` and `BNSA` were implemented to be applied to multiclass classification.
  - `AIRS` is more robust to noise in the data.
- **Optimization**: Use `Clonalg`.
  - The implementation can be applied to objective function optimization (min/max).
- **Clustering**: Use `AiNet`.
  - Automatically separates data into groups.
  - Does not require a predefined number of clusters.

---

### How do I normalize my data to use the `RNSA` algorithm?

RNSA works exclusively with data normalized in the range (0.0, 1.0). Therefore, before applying it, the data must be
normalized if they are not already in this range. A simple way to do this is by using normalization tools from
**scikit-learn**, such as
[``MinMaxScaler``](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

#### Example

In this example, `X` represents the non-normalized input data.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_norm = scaler.fit_transform(X)

# Training the model with normalized data
rnsa = RNSA(N=100, r=0.1)
rnsa.fit(x_norm, y)
```

---

## Parameter configuration

### How do I choose the number of detectors (`N`) in `RNSA` or `BNSA`?

The number of detectors directly affects performance:

- A small number of detectors may not adequately cover the non-self space.
- A very large number of detectors may increase training time and can cause overfitting.

**Recommendations**:

- Test different values for the number of detectors until you find a suitable balance between training time and model performance.
- Use cross-validation to identify the value that consistently yields the best results.

---

### Which radius (`r` or `aff_thresh`) should I use in `BNSA` or `RNSA`?

The detector radius depends on the data distribution:

- A very small radius may fail to detect anomalies.
- A very large radius may overlap the self space and never generate valid detectors.

---

### What is the `r_s` parameter in `RNSA`?

`r_s` is the radius of the self sample. It defines a region around each training sample.

---

### Clonalg: How do I define the objective function?

The objective function must follow the pattern of the
[base class](https://github.com/AIS-Package/aisp/blob/main/aisp/base/core/_optimizer.py#L140).
It must receive a solution as input and return a cost (or affinity) value.

```python
def affinity_function(self, solution: Any) -> float:
    pass
```

There are two ways to define the objective function in Clonalg.

```python
def sphere(solution):
    return np.sum(solution ** 2)
```

1. Defining the function directly in the class constructor

```python
clonalg = Clonalg(
    problem_size=2,
    affinity_function=sphere
)
```

2. Using the function registry

```python
clonalg = Clonalg(
    problem_size=2,
)

clonalg.register("affinity_function", sphere)
```

## Additional information

### Where can I find more examples?

- [Examples in the Jupyter Notebooks](../../examples/en).

### How can I contribute to the project?

See the [Contribution Guide](../../CONTRIBUTING.md) on GitHub.

### Still have questions?

- Open an [**Issue on GitHub**](https://github.com/AIS-Package/aisp/issues)
