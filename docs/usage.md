---
title: Basic usage
---
# `xplor`: Vectorized Optimization Modeling

Welcome to `xplor`. This guide shows you how to use familiar **polars DataFrame** operations to build optimization models, abstracting away the specific syntax of underlying solvers like **Gurobi** and **MathOpt**. Learn how to map your data directly into variables, constraints, and objective functions.

## Create an `xplor` Model

An `xplor` model is a **thin wrapper** around a classic solver model (e.g., `gurobi.Model`, `mathopt.Model`, etc.). This object allows us to **unify all solver-specific syntax** into a single **Polars-based syntax**.

```python
import xplor
from xplor.gurobi import XplorGurobi

# Initialize the xplor wrapper around a Gurobi model
xmodel = XplorGurobi() # or XplorMathOpt()
```

You can as well provide your own instance:

```python
import gurobipy as gp
model = gp.Model(name="MyModel", env=gp.Env())
xmodel = XplorGurobi(model)

# You can always access the underlying model with the `model` attribute
xmodel.model.Params.TimeLimit = 10.0
```

-----

## Creating Variables

In `xplor`, optimization variables are defined on a per-row basis within a Polars `DataFrame`. The `xmodel.add_vars()` method is called *within* a `.with_columns()` operation on a DataFrame.


### Example: Variables with Bounds and Objective Costs

Let's define a set of variables with unique lower bounds (`lb`), upper bounds (`ub`), and objective coefficients (`obj`).

```python
import polars as pl
import xplor
from xplor.types import VarType

data = pl.DataFrame({
    "product": ["P1", "P2"],
    "profit": [150.0, 200.0],
    "max_prod": [200.0, 150.0],
    "labor_hrs": [5.0, 10.0], # Labor hours per unit
    "material_weight": [4.0, 3.0], # Material pounds per unit

})

# Available Resources (Constant for constraints)
MAX_LABOR = 1500.0
MAX_WEIGHT = 1000.0

# Add variables to the DataFrame using xmodel.add_vars()
df = data.with_columns(xmodel.add_vars("x", lb="min_prod", ub="max_prod", obj="profit", indices=["product"]))
df
# shape: (2, 7)
# ┌─────────┬────────┬──────────┬───────────┬─────────────────┬────────────────────┐
# │ product ┆ profit ┆ max_prod ┆ labor_hrs ┆ material_weight ┆ x                  │
# │ ---     ┆ ---    ┆ ---      ┆ ---       ┆ ---             ┆ ---                │
# │ str     ┆ f64    ┆ f64      ┆ f64       ┆ f64             ┆ object             │
# ╞═════════╪════════╪══════════╪═══════════╪═════════════════╪════════════════════╡
# │ P1      ┆ 150.0  ┆ 200.0    ┆ 5.0       ┆ 4.0             ┆ <gurobi.Var x[P1]> │
# │ P2      ┆ 200.0  ┆ 150.0    ┆ 10.0      ┆ 3.0             ┆ <gurobi.Var x[P2]> │
# └─────────┴────────┴──────────┴───────────┴─────────────────┴────────────────────┘

```

> **Key Concept:** The `var()` call executes in the context of the Polars DataFrame, mapping the variable creation logic across every row. The variables themselves are stored internally in the `model` and returned as a Polars **Object Series**.

-----

## Variable expressions

```python
# You can defined expression by mixing `xplor.var` and polars expression
df = df.with_columns(labor_usage = xplor.var("x") * pl.col("labor_hrs"))
df
# shape: (2, 8)
# ┌─────────┬────────┬──────────┬───────────┬─────────────────┬────────────────────┬─────────────┐
# │ product ┆ profit ┆ max_prod ┆ labor_hrs ┆ material_weight ┆ x                  ┆ labor_usage │
# │ ---     ┆ ---    ┆ ---      ┆ ---       ┆ ---             ┆ ---                ┆ ---         │
# │ str     ┆ f64    ┆ f64      ┆ f64       ┆ f64             ┆ object             ┆ object      │
# ╞═════════╪════════╪══════════╪═══════════╪═════════════════╪════════════════════╪═════════════╡
# │ P1      ┆ 150.0  ┆ 200.0    ┆ 5.0       ┆ 4.0             ┆ <gurobi.Var x[P1]> ┆ 5.0 x[P1]   │
# │ P2      ┆ 200.0  ┆ 150.0    ┆ 10.0      ┆ 3.0             ┆ <gurobi.Var x[P2]> ┆ 10.0 x[P2]  │
# └─────────┴────────┴──────────┴───────────┴─────────────────┴────────────────────┴─────────────┘
```

-----

## Adding Constraints


The `xmodel.add_constrs()` method captures the symbolic expression, executes it in the context of the DataFrame, and adds the resulting constraints to the underlying solver model.

```python
df.select(
    xmodel.add_constrs(xplor.var("labor_usage").sum() <= MAX_LABOR, name="maxlabor"),
    xmodel.add_constrs((xplor.var("x") * pl.col("material_weight")).sum() <= MAX_WEIGHT) # name is deduce from the expression!
)
# shape: (1, 2)
# ┌─────────────────────────────┬─────────────────────────────────┐
# │ maxlabor                    ┆ (x * weight).sum() <= 1000.0    │
# │ ---                         ┆ ---                             │
# │ object                      ┆ object                          │
# ╞═════════════════════════════╪═════════════════════════════════╡
# │ <gurobi.Constr maxlabor[0]> ┆ <gurobi.Constr (x * weight).su… │
# └─────────────────────────────┴─────────────────────────────────┘
```

-----

## Solving the Model and Extracting Results

### Solving the Model

The `optimize()` method triggers the solver (Gurobi, in this case) to find the optimal solution.

```python
# Set the objective to Maximize (since obj is profit)
xmodel.model.setObjective(xmodel.model.getObjective(), sense=gp.GRB.MAXIMIZE)

# Solve the model
xmodel.optimize()
```

### Extracting the Objective Value

```python
xmodel.get_objective_value()
# 40000.0
```

### Extracting Variable Values

The optimal values for the variables are retrieved by referencing the **name** used when the variables were created (`"x"` in our example). The result is returned as a **Polars Series**.

```python
# Add solution back to the DataFrame
df.select("product", xmodel.get_variable_values("x").alias("production_units"))
# shape: (2, 2)
# ┌─────────┬──────────────────┐
# │ product ┆ production_units │
# │ ---     ┆ ---              │
# │ str     ┆ f64              │
# ╞═════════╪══════════════════╡
# │ P1      ┆ 200.0            │
# │ P2      ┆ 50.0             │
# └─────────┴──────────────────┘

```
