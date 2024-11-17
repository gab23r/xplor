# xplor: A Modern DataFrame-Centric Optimization Framework

[![PyPI version](https://badge.fury.io/py/xplor.svg)](https://badge.fury.io/py/xplor)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

xplor provides a unified framework for building Operation Research models using [polars](https://pola.rs/) DataFrames. By leveraging polars' performance and ergonomic API, xplor makes mathematical optimization more intuitive and maintainable.

## Features

- 🚀 **Polars Integration**: Built on top of polars for high-performance data operations
- 🧩 **Solver Agnostic**: Designed to support multiple solvers (currently Gurobi, more coming soon)
- 📝 **Intuitive API**: Natural expression syntax for constraints and objectives
- ⚡ **Vectorized Operations**: Efficient model building with DataFrame operations
- 🔍 **Type Hints**: Full typing support for better IDE integration

## Installation

```bash
pip install xplor
```

For Gurobi support, make sure you have Gurobi installed and licensed:
```bash
pip install gurobipy
```

## Quick Start

Here's a simple example showing how to build and solve an optimization model using xplor:

```python
>>> import xplor.gurobi as xpg
>>> import polars as pl
>>> import gurobipy as gp

# Wrap your model with XplorGurobi
>>> model = gp.Model()
>>> xmodel = xpg.XplorGurobi(model, deterministic=True, auto_update=True)

# Create sample data
>>> df = pl.DataFrame({
...     "i": [0, 0, 1, 2, 2],
...     "j": [1, 2, 0, 0, 1],
...     "u": [0.3, 1.2, 0.7, 0.9, 1.2],
...     "c": [1.3, 1.7, 1.4, 1.1, 0.9],
...     "obj": [2.5, 2.7, 1.2, 1.7, 3.9],
... })

# Add variables
>>> df = (
...     df
...     .pipe(xmodel.add_vars, name="x", ub="u", obj="obj", indices=["i", "j"])
...     .pipe(xpg.apply_eval, "y = 2 * x - c")
... )
>>> df
shape: (5, 7)
┌─────┬─────┬─────┬─────┬─────┬─────────────────────┬───────────────────┐
│ i   ┆ j   ┆ u   ┆ c   ┆ obj ┆ x                   ┆ y                 │
│ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ ---                 ┆ ---               │
│ i64 ┆ i64 ┆ f64 ┆ f64 ┆ f64 ┆ object              ┆ object            │
╞═════╪═════╪═════╪═════╪═════╪═════════════════════╪═══════════════════╡
│ 0   ┆ 1   ┆ 0.3 ┆ 1.3 ┆ 2.5 ┆ <gurobi.Var x[0,1]> ┆ -1.3 + 2.0 x[0,1] │
│ 0   ┆ 2   ┆ 1.2 ┆ 1.7 ┆ 2.7 ┆ <gurobi.Var x[0,2]> ┆ -1.7 + 2.0 x[0,2] │
│ 1   ┆ 0   ┆ 0.7 ┆ 1.4 ┆ 1.2 ┆ <gurobi.Var x[1,0]> ┆ -1.4 + 2.0 x[1,0] │
│ 2   ┆ 0   ┆ 0.9 ┆ 1.1 ┆ 1.7 ┆ <gurobi.Var x[2,0]> ┆ -1.1 + 2.0 x[2,0] │
│ 2   ┆ 1   ┆ 1.2 ┆ 0.9 ┆ 3.9 ┆ <gurobi.Var x[2,1]> ┆ -0.9 + 2.0 x[2,1] │
└─────┴─────┴─────┴─────┴─────┴─────────────────────┴───────────────────┘

# Add constraints using grouped operations
>>> (
...     df
...     .group_by("i")
...     .agg(xpg.quicksum("y"), pl.col("c").min())
...     .pipe(xmodel.add_constrs, "y <= c", name="sum_on_j", indices=["i"])
... )
shape: (3, 4)
┌─────┬────────────────────────────────┬─────┬─────────────────────────────┐
│ i   ┆ y                              ┆ c   ┆ sum_on_j                    │
│ --- ┆ ---                            ┆ --- ┆ ---                         │
│ i64 ┆ object                         ┆ f64 ┆ object                      │
╞═════╪════════════════════════════════╪═════╪═════════════════════════════╡
│ 1   ┆ -1.4 + 2.0 x[1,0]              ┆ 1.4 ┆ <gurobi.Constr sum_on_j[1]> │
│ 0   ┆ -3.0 + 2.0 x[0,1] + 2.0 x[0,2] ┆ 1.3 ┆ <gurobi.Constr sum_on_j[0]> │
│ 2   ┆ -2.0 + 2.0 x[2,0] + 2.0 x[2,1] ┆ 0.9 ┆ <gurobi.Constr sum_on_j[2]> │
└─────┴────────────────────────────────┴─────┴─────────────────────────────┘

# Solve the model
>>> model.optimize()

# Extract solution
>>> df.with_columns(xpg.read_value("x"))
shape: (5, 7)
┌─────┬─────┬─────┬─────┬─────┬─────┬───────────────────┐
│ i   ┆ j   ┆ u   ┆ c   ┆ obj ┆ x   ┆ y                 │
│ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ ---               │
│ i64 ┆ i64 ┆ f64 ┆ f64 ┆ f64 ┆ f64 ┆ object            │
╞═════╪═════╪═════╪═════╪═════╪═════╪═══════════════════╡
│ 0   ┆ 1   ┆ 0.3 ┆ 1.3 ┆ 2.5 ┆ 0.0 ┆ -1.3 + 2.0 x[0,1] │
│ 0   ┆ 2   ┆ 1.2 ┆ 1.7 ┆ 2.7 ┆ 0.0 ┆ -1.7 + 2.0 x[0,2] │
│ 1   ┆ 0   ┆ 0.7 ┆ 1.4 ┆ 1.2 ┆ 0.0 ┆ -1.4 + 2.0 x[1,0] │
│ 2   ┆ 0   ┆ 0.9 ┆ 1.1 ┆ 1.7 ┆ 0.0 ┆ -1.1 + 2.0 x[2,0] │
│ 2   ┆ 1   ┆ 1.2 ┆ 0.9 ┆ 3.9 ┆ 0.0 ┆ -0.9 + 2.0 x[2,1] │
└─────┴─────┴─────┴─────┴─────┴─────┴───────────────────┘
```

## Current Status

xplor is in active development. Currently supported:
- ✅ Gurobi backend
- ✅ Basic model building operations
- ✅ Variable and constraint creation
- ✅ Expression evaluation
- ✅ Solution reading

Planned features:
- 🚧 Support for additional solvers (CPLEX, CBC, SCIP)
- 🚧 Extended modeling capabilities
- 🚧 Performance optimizations
- 🚧 More utility functions

## Why xplor?

xplor aims to modernize the Operation Research workflow by:
1. Using polars instead of pandas for better performance and memory usage
2. Providing a consistent API across different solvers
3. Making model building more intuitive with DataFrame operations
4. Enabling better code organization and maintenance

## Comparison with Other Tools

xplor is heavily inspired by [gurobipy-pandas](https://github.com/Gurobi/gurobipy-pandas) but differs in these key aspects:
- Uses polars instead of pandas for better performance
- Designed to be solver-agnostic from the ground up

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

```bash
# Clone the repository
git clone https://github.com/gab23r/xplor.git
cd xplor

# Install development dependencies
uv sync --all-extras

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [gurobipy-pandas](https://github.com/Gurobi/gurobipy-pandas) for inspiration
- [polars](https://pola.rs/) for the amazing DataFrame library
- [Gurobi](https://www.gurobi.com/) for the optimization solver