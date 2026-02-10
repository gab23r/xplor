# API Reference

This section contains the complete API reference for xplor.

## Core Classes

The main model classes that provide solver-specific implementations:

- **[XplorGurobi](XplorGurobi.md)** - Gurobi solver backend
- **[XplorMathOpt](XplorMathOpt.md)** - OR-Tools MathOpt backend
- **[XplorHexaly](XplorHexaly.md)** - Hexaly solver backend
- **[XplorCplex](XplorCplex.md)** - CPLEX solver backend for mathematical programming (via docplex)
- **[XplorCplexCP](XplorCplexCP.md)** - CPLEX CP Optimizer backend for constraint programming and scheduling (via docplex.cp)


## Functions

- **[var](var.md)** - Create variable expressions in Polars DataFrames
