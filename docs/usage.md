# `xplor`: Vectorized Optimization Modeling

Welcome to `xplor`. This guide shows you how to use familiar **polars DataFrame** operations to build optimization models, abstracting away the specific syntax of underlying solvers like **Gurobi** and **MathOpt**. Learn how to map your data directly into variables, constraints, and objective functions.

## Create an `xplor` Model

An `xplor` model is a **thin wrapper** around a classic solver model (e.g., `gurobi.Model`, `mathopt.Model`, etc.). This object allows us to **unify all solver-specific syntax** into a single **Polars-based syntax**.

```python
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

## Creating Variables (`xmodel.var()`)

In `xplor`, optimization variables are defined on a per-row basis within a Polars `DataFrame`. The `xmodel.var()` method is called *within* a `.with_columns()` operation on a DataFrame.


### Example: Variables with Bounds and Objective Costs

Let's define a set of variables with unique lower bounds (`lb`), upper bounds (`ub`), and objective coefficients (`obj`).

```python
import polars as pl
from xplor.types import VarType

data = pl.DataFrame({
    "index_id": [1, 2, 3],
    "lb": [0.0, 1.0, 0.0],
    "ub": [10.0, 5.0, 20.0],
    "obj": [2.0, -1.5, 3.0],
})

# Add variables to the DataFrame using xmodel.var()
df = data.with_columns(
    xmodel.var(
        "x",
        lb="lb",
        ub="ub",
        obj="obj",
        indices=["index_id"],
        vtype=VarType.CONTINUOUS,
    )
)
df
# shape: (3, 5)
# ┌──────────┬─────┬──────┬──────┬───────────────────┐
# │ index_id ┆ lb  ┆ ub   ┆ obj  ┆ x                 │
# │ ---      ┆ --- ┆ ---  ┆ ---  ┆ ---               │
# │ i64      ┆ f64 ┆ f64  ┆ f64  ┆ object            │
# ╞══════════╪═════╪══════╪══════╪═══════════════════╡
# │ 1        ┆ 0.0 ┆ 10.0 ┆ 2.0  ┆ <gurobi.Var x[1]> │
# │ 2        ┆ 1.0 ┆ 5.0  ┆ -1.5 ┆ <gurobi.Var x[2]> │
# │ 3        ┆ 0.0 ┆ 20.0 ┆ 3.0  ┆ <gurobi.Var x[3]> │
# └──────────┴─────┴──────┴──────┴───────────────────┘

```

> **Key Concept:** The `var()` call executes in the context of the Polars DataFrame, mapping the variable creation logic across every row. The variables themselves are stored internally in the `model` and returned as a Polars **Object Series**.

-----

## Adding Constraints (`xmodel.constr()`)


The `xmodel.constr()` method captures the symbolic expression, executes it in the context of the DataFrame, and adds the resulting constraints to the underlying solver model.

### 1\. Simple constraint

To add a constraint for every row of the DataFrame (e.g., $x_i \le 5 \cdot \text{lb}_i$):

```python
df.select(xmodel.constr(xplor.var("x") <= 5 * pl.col("lb"), name="c"))
# shape: (3, 1)
# ┌──────────────────────────────┐
# │ c                    │
# │ ---                          │
# │ object                       │
# ╞══════════════════════════════╡
# │ <gurobi.Constr c[0]> │
# │ <gurobi.Constr c[1]> │
# │ <gurobi.Constr c[2]> │
# └──────────────────────────────┘
```

### 2\. Expression

You can used of polars expressiom syntax to build you constraint (e.g., $\sum_{i} (x_{i} - \min(\text{lb})) \le 15$):

```python
df.select(xmodel.constr((xplor.var("x") - pl.col("lb").min()).sum() <= 15.0)) # name of the constraint is deduced.
# shape: (1, 1)
# ┌─────────────────────────────────┐
# │ (x - lb.min()).sum() <= 15      │
# │ ---                             │
# │ object                          │
# ╞═════════════════════════════════╡
# │ <gurobi.Constr (x - lb.min()).… │
# └─────────────────────────────────┘
```

-----

## Solving the Model and Extracting Results

### Solving the Model

The `optimize()` method triggers the solver (Gurobi, in this case) to find the optimal solution.

```python
xmodel.optimize()
# Gurobi Optimizer version...
# ...
# Optimal objective -7.500000000e+00
```

### Extracting the Objective Value

```python
xmodel.get_objective_value()
# -7.5
```

### Extracting Variable Values

The optimal values for the variables are retrieved by referencing the **name** used when the variables were created (`"x"` in our example). The result is returned as a **Polars Series**.

```python
# Add solution back to the DataFrame based on index
data.with_columns(xmodel.get_variable_values("x"))
# shape: (3, 5)
# ┌──────────┬─────┬──────┬──────┬─────┐
# │ index_id ┆ lb  ┆ ub   ┆ obj  ┆ x   │
# │ ---      ┆ --- ┆ ---  ┆ ---  ┆ --- │
# │ i64      ┆ f64 ┆ f64  ┆ f64  ┆ f64 │
# ╞══════════╪═════╪══════╪══════╪═════╡
# │ 1        ┆ 0.0 ┆ 10.0 ┆ 2.0  ┆ 0.0 │
# │ 2        ┆ 1.0 ┆ 5.0  ┆ -1.5 ┆ 5.0 │
# │ 3        ┆ 0.0 ┆ 20.0 ┆ 3.0  ┆ 0.0 │
# └──────────┴─────┴──────┴──────┴─────┘
```
