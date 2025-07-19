PKG = aisp

.PHONY: dev test lint docstyle check

# Installs development dependencies to set up the environment
dev:
	pip install -e .[dev]

# Runs tests with detailed output
test:
	pytest $(PKG) -v

# Runs pylint for linting on the package
lint:
	pylint $(PKG) -v

# Checks docstring style with pydocstyle (numpydoc convention)
docstyle:
	pydocstyle $(PKG) -v

# Runs all quality checks
check:
	$(MAKE) lint
	$(MAKE) docstyle
	$(MAKE) test
