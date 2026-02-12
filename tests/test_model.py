import polars as pl
import pytest
from ortools.math_opt.python import mathopt

import xplor
from xplor.gurobi import XplorGurobi
from xplor.mathopt import XplorMathOpt
from xplor.model import XplorModel
from xplor.types import VariableType


@pytest.mark.parametrize(
    ("ModelClass", "solver_type", "vtype", "obj_value", "var_value", "dtype"),
    [
        (XplorGurobi, None, "CONTINUOUS", -2.5, [0.5, 1.0], pl.Float64),
        # (XplorHexaly, None, "CONTINUOUS", -2.5, [0.5, 1.0], pl.Float64),
        (XplorMathOpt, mathopt.SolverType.GLOP, None, -2.5, [0.5, 1.0], pl.Float64),
        (XplorMathOpt, mathopt.SolverType.CP_SAT, "INTEGER", -2.0, [0, 1], pl.Int32),
        (XplorMathOpt, mathopt.SolverType.CP_SAT, "BINARY", -2.0, [False, True], pl.Boolean),
    ],
)
def test_linear_optimization_problem(
    ModelClass: type[XplorModel],
    solver_type: mathopt.SolverType | None,
    vtype: VariableType,
    obj_value: float,
    var_value: list[float],
    dtype: pl.DataType,
):
    """
    Tests a basic linear optimization problem using different solver wrappers
    (Gurobi, MathOpt) to ensure they yield the same result.
    """
    xmodel = ModelClass()  # type: ignore

    df = pl.DataFrame(
        {"id": [0, 1], "lb": [-1.0, 0.0], "ub": [1.5, 1.0], "obj": [-1, -2]}
    ).with_columns(xmodel.add_vars("x", lb="lb", ub="ub", obj="obj", vtype=vtype))
    df.pipe(xmodel.add_constrs, xplor.var.x.sum() <= 1.5)
    if solver_type is not None:
        xmodel.optimize(solver_type=solver_type)
    else:
        xmodel.optimize()

    # Check Objective Value
    objective_value = xmodel.get_objective_value()
    assert objective_value == pytest.approx(obj_value, rel=1e-4)

    # Check Variable Values
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result.dtype == dtype
    assert result.to_list() == pytest.approx(var_value, rel=1e-4)


# @pytest.mark.skip("need to add xmodel.constrs attributes")
def test_add_constrs():
    xmodel = XplorGurobi()

    df = pl.DataFrame(
        {"id": [0, 1], "lb": [-1.0, 0.0], "ub": [1.5, 1.0], "obj": [-1, -2]}
    ).with_columns(xmodel.add_vars("x"), xmodel.add_vars("y"))

    xmodel.add_constrs(df, xplor.var("x", "y").sum().name.suffix(".sum()") == 1)
    xmodel.add_constrs(df, (xplor.var("x") + 1).sum() >= 1.5, (xplor.var("y") + 1).sum() >= 1.5)
    xmodel.add_constrs(
        df,
        xplor.var("x") - xplor.var("y") >= 0.5,
    )
    xmodel.model.update()
    assert [constr.ConstrName for constr in xmodel.model.getConstrs()] == [
        "x.sum()[0]",
        "y.sum()[0]",
        "(x + 1).sum() >= 1.5",
        "(y + 1).sum() >= 1.5",
        "(x - y) >= 0.5[0]",
        "(x - y) >= 0.5[1]",
    ]


def test_general_constraint():
    """Test Gurobi General Constraints (GenConstr) with max."""
    import gurobipy as gp

    xmodel = XplorGurobi()

    # Create binary variables with negative objective (to maximize we minimize negative)
    df = pl.DataFrame(height=3).with_columns(xmodel.add_vars("x", vtype="BINARY", obj=-1.0))

    # Create a GenExpr for max using gp.max_ directly
    x_vars = df["x"].to_list()
    max_var = gp.max_(x_vars)  # This creates a GenExpr

    # Broadcast the GenExpr to all rows using pl.Series
    df = df.with_columns(m=pl.Series([max_var] * len(df), dtype=pl.Object))

    # Add constraint that x == m (this creates general constraints, one per row)
    xmodel.add_constrs(df, a=xplor.var.x == xplor.var.m)

    # Minimize negative objective (equivalent to maximizing)
    xmodel.optimize()

    # The solution: since each x == max(all x), and we maximize sum(x), all x should be 1
    result = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert result.sum() == 3  # All should be 1


def test_general_constraint_min():
    """Test Gurobi General Constraints with min."""
    import gurobipy as gp

    xmodel = XplorGurobi()

    # Create continuous variables with different lower bounds
    df = pl.DataFrame({"id": [0, 1, 2], "lb": [1.0, 2.0, 3.0]}).with_columns(
        xmodel.add_vars("x", lb="lb", ub=10, obj=1.0)
    )

    # Create an auxiliary variable to hold the min value
    df_single = pl.DataFrame(height=1).with_columns(xmodel.add_vars("min_x", lb=0, ub=10))

    # Create a GenExpr for min
    x_vars = df["x"].to_list()
    min_var = gp.min_(x_vars)

    # Add constraint: min_x == min(all x)
    df_single = df_single.with_columns(min_expr=pl.lit(min_var, allow_object=True))
    xmodel.add_constrs(df_single, min_constr=xplor.var.min_x == xplor.var.min_expr)

    # Minimize sum of x
    xmodel.optimize()

    # All x should be at their lower bounds (1.0, 2.0, 3.0), so min should be 1.0
    x_values = df.select(xmodel.read_values(pl.col("x"))).to_series()
    assert x_values.to_list() == pytest.approx([1.0, 2.0, 3.0], rel=1e-4)


def test_general_constraint_abs():
    """Test Gurobi General Constraints with abs."""
    import gurobipy as gp

    xmodel = XplorGurobi()

    # Create variables
    df = pl.DataFrame({"id": [0, 1], "value": [-2.0, 3.0]}).with_columns(
        xmodel.add_vars("x", lb=-5, ub=5), xmodel.add_vars("y", lb=0, ub=10)
    )

    # Create GenExpr for absolute value for each x variable
    df = df.with_columns(abs_x=df["x"].map_elements(lambda x: gp.abs_(x), return_dtype=pl.Object))

    # Add constraints: x = value and y = abs_x
    xmodel.add_constrs(df, c=xplor.var.x == pl.col("value"))
    xmodel.add_constrs(df, d=xplor.var.y == xplor.var.abs_x)

    xmodel.optimize()

    # Check that y values equal the absolute values of the value column
    y_values = df.select(xmodel.read_values(pl.col("y"))).to_series()
    expected = [2.0, 3.0]
    assert y_values.to_list() == pytest.approx(expected, rel=1e-4)
