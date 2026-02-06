# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xplor is a DataFrame-centric optimization framework that provides a unified Polars-based API for building Operation Research models across multiple solver backends (Gurobi, OR-Tools, Hexaly). The framework abstracts solver-specific syntax, enabling users to define a single model and benchmark performance across different solvers.

**Key Principle**: Model logic is expressed through vectorized Polars DataFrame operations rather than Python loops. While variable/constraint objects themselves are Python objects (e.g., `gurobi.Var`), data manipulation happens in Rust via Polars UDFs for performance.

## Development Commands

Uses `uv` for package management and `just` for task running.

### Common Commands
```bash
# Install all dependencies (including optional solver backends)
just install

# Run all tests
just test

# Run tests with coverage report
just test-cov

# Lint and format check
just lint

# Auto-fix linting and formatting issues
just fix

# Type checking
just ty

# Run pre-commit hooks
just prek

# Build package
just build

# Build documentation
just docs-build

# Serve docs locally with live reload
just docs-serve

# Deploy docs to GitHub Pages
just docs-deploy

# Clean build artifacts
just clean
```

### Testing Specific Tests
```bash
# Run a specific test file
uv run pytest tests/test_model.py

# Run a specific test function
uv run pytest tests/test_model.py::test_function_name

# Run tests matching a pattern
uv run pytest -k "pattern"
```

## Architecture

### Core Abstraction Layer

- **`src/xplor/model.py`**: Defines `XplorModel`, the abstract base class that all solver backends inherit from. This class provides the unified interface for:
  - `add_vars()`: Create optimization variables via Polars expressions
  - `add_constrs()`: Add constraints via Polars expressions
  - `optimize()`: Solve the model
  - `read_values()`: Extract solution values back into DataFrames
  - `get_objective_value()`: Retrieve objective value

### Solver Backend Implementations

Each backend implements the abstract methods from `XplorModel`:

- **`src/xplor/gurobi/model.py`**: `XplorGurobi` - Gurobi solver backend
- **`src/xplor/mathopt/model.py`**: `XplorMathOpt` - OR-Tools MathOpt backend
- **`src/xplor/hexaly/model.py`**: `XplorHexaly` - Hexaly solver backend

All backends translate the abstract Polars-based operations into solver-specific API calls.

### Expression System

The `src/xplor/exprs/` module handles symbolic mathematical expressions:

- **`var.py`**: Variable expressions (`_ProxyVarExpr` provides the `xplor.var()` API)
- **`obj.py`**: Objective function expressions and evaluation logic
- **`constr.py`**: Constraint expressions (equality, inequality, range constraints)

Expressions are lazy - they produce Polars expressions that are only evaluated when materialized (e.g., via `.with_columns()` or `.select()`).

### Key Design Patterns

1. **Polars Expression-Based API**: Variables and constraints are created via Polars expressions rather than imperative loops. This enables vectorized operations.

2. **Lazy Evaluation**: `add_vars()` returns a Polars expression that only creates variables when consumed by DataFrame operations.

3. **Backend Agnosticism**: The same model definition works across all solver backends by swapping the wrapper class (e.g., `XplorGurobi` → `XplorMathOpt`).

4. **Variable Storage**: Created variables are stored in `XplorModel.vars` (dict mapping name → Polars Series of solver variable objects) and `XplorModel.var_types` (dict mapping name → VarType).

## Package Management

- Uses `uv` (not pip/poetry/conda)
- Python 3.11+ required
- Optional dependency groups: `gurobi`, `ortools`, `hexaly`, `all`
- Special index for Hexaly packages (https://pip.hexaly.com)

## Code Quality Standards

### Linting and Formatting
- **Ruff** for both linting and formatting (see `pyproject.toml` for extensive rule configuration)
- Line length: 100 characters
- Target: Python 3.11

### Type Checking
- Uses `ty` (not mypy) for type checking
- Run with `just ty` or `uv run ty check .`

### Pre-commit Hooks
- Managed via `prek` (not standard pre-commit)
- Hooks include: trailing whitespace, file size checks, ruff, uv lock, ty, pytest
- Install/run: `just prek`

### Testing
- Uses pytest
- Test files in `tests/`
- Coverage reports available via `just test-cov`

## Documentation

- Built with **MkDocs** using **Material for MkDocs** theme
- API documentation generated automatically with **mkdocstrings** from NumPy-style docstrings
- Source: `docs/` directory (Markdown files)
- Home page (`docs/index.md`) includes content from `README.md` via snippets
- Published to: https://gab23r.github.io/xplor/
- Build output: `site/` directory (auto-generated, do not modify)
- Local preview: `just docs-serve` (live reload at http://127.0.0.1:8000)
- Deploy: `just docs-deploy` or via GitHub Actions on push to main

## Important Notes

- **Variable Types**: Defined in `src/xplor/types.py` as `VarType` enum (CONTINUOUS, INTEGER, BINARY)
- **Solver Dependencies**: Optional - install specific backends as needed
- **Testing Multi-Backend**: Tests should work across all available backends when possible
- **Expression Parsing**: The `add_constrs()` method handles both row-wise and aggregated constraints differently (see docstring warnings about granularity)
