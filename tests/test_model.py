import polars as pl
import pytest
from ortools.math_opt.python import mathopt

import xplor
from xplor.gurobi import XplorGurobi
from xplor.mathopt import XplorMathOpt
from xplor.model import XplorModel
from xplor.types import VarType


@pytest.mark.parametrize(
    ("ModelClass", "solver_type", "vtype", "obj_value", "var_value", "dtype"),
    [
        (XplorGurobi, None, VarType.CONTINUOUS, -2.5, [0.5, 1.0], pl.Float64),
        # (XplorHexaly, None, VarType.CONTINUOUS, -2.5, [0.5, 1.0], pl.Float64),
        (XplorMathOpt, mathopt.SolverType.GLOP, None, -2.5, [0.5, 1.0], pl.Float64),
        (XplorMathOpt, mathopt.SolverType.CP_SAT, VarType.INTEGER, -2.0, [0, 1], pl.Int32),
        (XplorMathOpt, mathopt.SolverType.CP_SAT, VarType.BINARY, -2.0, [False, True], pl.Boolean),
    ],
)
def test_linear_optimization_problem(
    ModelClass: type[XplorModel],
    solver_type: mathopt.SolverType | None,
    vtype: VarType | None,
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
    df.select(xmodel.add_constrs(xplor.var.x.sum() <= 1.5))
    if solver_type is not None:
        xmodel.optimize(solver_type=solver_type)
    else:
        xmodel.optimize()

    # Check Objective Value
    objective_value = xmodel.get_objective_value()
    assert objective_value == pytest.approx(obj_value, rel=1e-4)

    # Check Variable Values
    result = xmodel.get_variable_values("x")
    assert result.dtype == dtype
    assert result.to_list() == pytest.approx(var_value, rel=1e-4)
