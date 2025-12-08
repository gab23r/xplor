# xplor: A Modern DataFrame-Centric Optimization Framework

[![PyPI version](https://badge.fury.io/py/xplor.svg)](https://badge.fury.io/py/xplor)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

xplor provides a unified framework for building Operation Research models using [polars](https://pola.rs/) DataFrames. By leveraging polars' performance and ergonomic API, xplor makes mathematical optimization more intuitive and maintainable.

## Why xplor?


Building large-scale optimization models in Python can become **inefficient**. Standard approaches often require iterating over rows, manipulating Python dictionaries, or using slow list comprehensions, which makes the logic **slow and difficult to maintain** as the model grows.


**xplor** solves this core issue by using a **Polars-based DataFrame API** to construct your model logic. This provides two significant benefits:

1.  **Vectorized Performance:** Model operations (like grouping, joining, filtering, and aggregation) are executed using **Polars' high-performance Rust backend**. You are no longer manipulating slow Python objects*; the logic is executed at native speed.
2.  **Maintainability:** By relying on the familiar and expressive DataFrame API, your model logic becomes significantly **more readable and maintainable**.

Additionally, a huge benefit of `xplor` is its **solver agnosticism**. It abstracts away the specific syntax of underlying optimization engines (like Gurobi or MathOpt), allowing you to define a single model and easily benchmark its performance across multiple solvers.

\* Clarification: Variable, variable expression, and constraint objects themselves are still Python objects (e.g., `gurobi.Var` or `mathopt.Variable`), but the data manipulation and application of model logic are managed by Rust via Polars' User-Defined Functions (UDFs), enabling vectorized performance.


## Installation

**`xplor`** requires Python 3.11+.

Install the base package, which provides the core Polars integration and abstract modeling interface:

xplor comes with few dependencies. To use a specific solver you need to add optional dependency group.
```bash
pip install xplor[gurobi, ortools]
```
or install all of them
```bash
pip install xplor[all]
```

> **Note:** Gurobi requires a a valid license (academic or commercial) to run. Please follow the [official Gurobi documentation](https://docs.gurobi.com/current/) for setup instructions.

## Quick Start

Have a look to the [documentation](https://gab23r.github.io/xplor/usage.html).

## 🟢 Current Status

`xplor` is in active development.

### Currently Supported Backends

* ✅ **Gurobi backend**
* ✅ **OR-Tools backend**
* ✅ **Hexaly backend**

### Planned Features

* 🚧 Support for additional solvers (CPLEX, ...).
* 🚧 Support for non-linear expressions.

## Contributing

Want to contribute? Read our [contributing guide](https://gab23r.github.io/xplor/contributing.html).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [gurobipy-pandas](https://github.com/Gurobi/gurobipy-pandas) for inspiration.
- [polars](https://pola.rs/) for the amazing DataFrame library.
- [Gurobi](https://www.gurobi.com/) for the optimization solver.
- [Ortools](https://developers.google.com/optimization/math_opt) for the optimization solver.
- [great-docs](https://github.com/rich-iannone/great-docs) for generating the docs.
