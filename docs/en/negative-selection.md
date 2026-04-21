---
id: docs-nsa
keywords:
    - negative selection
    - nsa
    - artificial immune systems
    - anomaly detection
    - classification
    - bio-inspired algorithms
    - natural computing
---

# Seleção Negativa

Os algoritmos de **seleção negativa** is the process in which the immune system maturates T-cells, also known as
T-lymphocytes, which make them capable of detecting non-self. Thus, the Negative Selection Algorithm (NSA)
uses hyperspheres symbolizing the detectors in an N-dimensional data space. [^1]

---

Negative Selection can be applied in different contexts, such as:
- **Anomaly detection**
- **Classification**

## Package implementation

### Binary Negative Selection Algorithm ([BNSA](./api/nsa/bnsa.md))

The binary algorithm adapted for multiple classes in this project is based on the version proposed by
Forrest et al. (1994)[^2], originally developed for computer security.

### Real-Valued Negative Selection Algorithm ([RNSA](./api/nsa/rnsa.md))

This algorithm has two different versions: one based on the canonical version [^1] and another with variable
radius detectors.[^3] Both are adapted to work with multiple classes and have methods for predicting data
present in the non-self region of all detectors and classes.

## References

[^1]: BRABAZON, Anthony; O'NEILL, Michael; MCGARRAGHY, Seán. Natural Computing
    Algorithms. [S. l.]: Springer Berlin Heidelberg, 2015. DOI 10.1007/978-3-662-43631-8.
    Available at: https://dx.doi.org/10.1007/978-3-662-43631-8.

[^2]: S. Forrest, A. S. Perelson, L. Allen and R. Cherukuri, "Self-nonself discrimination in
    a computer," Proceedings of 1994 IEEE Computer Society Symposium on Research in Security
    and Privacy, Oakland, CA, USA, 1994, pp. 202-212,
    doi: https://dx.doi.org/10.1109/RISP.1994.296580.

[^3] Ji, Z.; Dasgupta, D. (2004). Real-Valued Negative Selection Algorithm with Variable-Sized Detectors.
    In *Lecture Notes in Computer Science*, vol. 3025. https://doi.org/10.1007/978-3-540-24854-5_30
