PKG = aisp

.PHONY: install install-dev dev test lint docstyle check

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Installs development dependencies to set up the environment
dev: install-dev
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
