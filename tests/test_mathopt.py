import polars as pl

import xplor
from xplor.mathopt import XplorMathOpt


def test_mathopt_model() -> None:
    xmodel = XplorMathOpt()
    (
        pl.DataFrame({"lb": [-1.0, 0.0], "ub": [1.5, 1.0], "obj": [-1, -2]})
        .with_columns(xmodel.var("x", lb="lb", ub="ub", obj="obj"))
        .select(xplor.var.x.sum())
        .with_columns(xmodel.constr(xplor.var.x <= 1.5))
    )
    xmodel.optimize()
    assert xmodel.get_objective_value() == -2.5
