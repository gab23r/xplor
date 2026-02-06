# API Reference

This section contains the complete API reference for xplor.

## Core Classes

The main model classes that provide solver-specific implementations:

- **[XplorGurobi](XplorGurobi.md)** - Gurobi solver backend
- **[XplorMathOpt](XplorMathOpt.md)** - OR-Tools MathOpt backend
- **[XplorHexaly](XplorHexaly.md)** - Hexaly solver backend

## Types

- **[VarType](VarType.md)** - Enumeration for variable types (CONTINUOUS, INTEGER, BINARY)

## Functions

- **[var](var.md)** - Create variable expressions in Polars DataFrames
