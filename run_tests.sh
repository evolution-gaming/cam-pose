#!/bin/bash
# Run tests and check code quality

# Execute unit tests using pytest
pytest

# Run Flake8 to check for Python style and linting issues
flake8 *.py src/ tests/

# Check if imports are properly sorted according to project standards
isort . --check --diff
