# Contributing to AISP

Thank you for considering contributing to AISP(Artificial Immune Systems Package)!

This project is maintained by small team, which means that new feature, especially complex algorithms, may take time
for study, development, and review. Therefore, your help is immensely appreciated.

There are several ways to contribute:

- Reporting bugs or unexpected behavior.
- Suggesting or implementing new immune-inspired algorithms.
- Most importantly, helping with documentation and examples: [see more details here](https://github.com/AIS-Package/aisp/issues/42)

## Project Standards

To maintain consistency across the package, please follow these guidelines:

### Docstrings

- All files and functions must include docstrings.
- The docstring format follows the [Numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)

### Architecture and Style

- The package is modular and organized into submodules, with algorithms distributed to their domain within artificial immune systems.
- Use English for variable and function names.
- Prefer focused functions, avoid monolithic functions.
- Whenever possible, include type annotations to improbe usability.
- Include automated test for any new functionality added.

### Formatting and Quality

- Run the make checks, especially ``make check``, and optionally ``make typecheck`` for static type checking.
- Update the documentation if any dependencies are added.

## Acknowledgment

Thank you very much for dedicating your time and talent to AISP! Every contribution helps make the package more robust.
