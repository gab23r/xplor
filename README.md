## xplor - Operation Research with polars


This package is heavily inspired by [gurobipy-pandas](https://github.com/Gurobi/gurobipy-pandas) but uses polars as dataframe backend.


### Basic example
```python

import xplor.gurobi as pg
import gurobipy as gp

model = gp.Model()

df = pl.DataFrame(
    {
        "i": [0, 0, 1, 2, 2],
        "j": [1, 2, 0, 0, 1],
        "u": [0.3, 1.2, 0.7, 0.9, 1.2],
        "c": [1.3, 1.7, 1.4, 1.1, 0.9],
        "obj": [2.5, 2.7, 1.2, 1.7, 3.9],
    }
)

df = (
    df
    .pipe(pg.add_vars, model, name="x", ub="u", obj = "obj", indices = ["i", "j"], vtype = gp.GRB.CONTINUOUS)
)
# shape: (5, 6)
# ┌─────┬─────┬─────┬─────┬─────┬─────────────────────┐
# │ i   ┆ j   ┆ u   ┆ c   ┆ obj ┆ x                   │
# │ --- ┆ --- ┆ --- ┆ --- ┆ --- ┆ ---                 │
# │ i64 ┆ i64 ┆ f64 ┆ f64 ┆ f64 ┆ object              │
# ╞═════╪═════╪═════╪═════╪═════╪═════════════════════╡
# │ 0   ┆ 1   ┆ 0.3 ┆ 1.3 ┆ 2.5 ┆ <gurobi.Var x[0,1]> │
# │ 0   ┆ 2   ┆ 1.2 ┆ 1.7 ┆ 2.7 ┆ <gurobi.Var x[0,2]> │
# │ 1   ┆ 0   ┆ 0.7 ┆ 1.4 ┆ 1.2 ┆ <gurobi.Var x[1,0]> │
# │ 2   ┆ 0   ┆ 0.9 ┆ 1.1 ┆ 1.7 ┆ <gurobi.Var x[2,0]> │
# │ 2   ┆ 1   ┆ 1.2 ┆ 0.9 ┆ 3.9 ┆ <gurobi.Var x[2,1]> │
# └─────┴─────┴─────┴─────┴─────┴─────────────────────┘

(
    df
    .pipe(pg.apply_eval, "y = 2 * x - c")
    .group_by("i").agg(pg.quicksum("y"), pl.col("c").min())
    .pipe(pg.add_constrs, model, "y <= c", name="constr")
)
# shape: (3, 4)
# ┌─────┬────────────────────────────────┬─────┬────────────────────────┐
# │ i   ┆ y                              ┆ c   ┆ constr                 │
# │ --- ┆ ---                            ┆ --- ┆ ---                    │
# │ i64 ┆ object                         ┆ f64 ┆ object                 │
# ╞═════╪════════════════════════════════╪═════╪════════════════════════╡
# │ 1   ┆ -1.4 + 2.0 x[1,0]              ┆ 1.4 ┆ <gurobi.Constr constr> │
# │ 0   ┆ -3.0 + 2.0 x[0,1] + 2.0 x[0,2] ┆ 1.3 ┆ <gurobi.Constr constr> │
# │ 2   ┆ -2.0 + 2.0 x[2,0] + 2.0 x[2,1] ┆ 0.9 ┆ <gurobi.Constr constr> │
# └─────┴────────────────────────────────┴─────┴────────────────────────┘

model.optimize()
```