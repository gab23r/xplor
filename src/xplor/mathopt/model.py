from __future__ import annotations

import polars as pl
from ortools.math_opt.python import mathopt, result

from xplor._utils import map_rows
from xplor.model import XplorModel
from xplor.var import VarType


class XplorMathOpt(XplorModel):
    """Xplor base class to wrap your MathOpt model."""

    model: mathopt.Model
    result: result.SolveResult

    def __init__(self, model: mathopt.Model | None = None) -> None:
        """Initialize the model wrapper.

        Concrete implementations must handle model initialization and setup.
        """
        model = mathopt.Model() if model is None else model
        super().__init__(model=model)

    def _add_vars(
        self,
        df: pl.DataFrame,
        vtype: VarType = VarType.CONTINUOUS,
    ) -> pl.Series:
        """Return a series of variables.

        `df` should contains columns: ["lb", "ub", "obj", "name"].
        """
        # mathopt.Model don't super binary variable directly
        if vtype == "BINARY":
            df = df.with_columns(
                pl.col("lb").clip(lower_bound=0).fill_null(0),
                pl.col("ub").clip(upper_bound=1).fill_null(1),
            )

        series = pl.Series(
            [
                self.model.add_variable(
                    lb=lb_, ub=ub_, name=name_, is_integer=vtype != VarType.CONTINUOUS
                )
                for lb_, ub_, name_ in df.drop("obj").rows()
            ],
            dtype=pl.Object,
        )
        if df.select("obj").filter(pl.col("obj") != 0).height:
            self.model.minimize_linear_objective(
                sum([w * v for w, v in zip(df["obj"], series, strict=True)])
            )

        return series

    def _add_constrs(self, df: pl.DataFrame, name: str, expr_str: str) -> pl.Series:
        """Return a series of variables.

        `df` should contains columns: ["lb", "ub", "name"].
        """
        # TODO: manage non linear constraint
        # https://github.com/gab23r/xplor/issues/1

        return map_rows(
            df,
            lambda d: self.model.add_linear_constraint(eval(expr_str), name=name),
        )

    def optimize(self) -> None:
        """Solve the model."""
        self.result = mathopt.solve(self.model, mathopt.SolverType.GLOP)

    def get_objective_value(self) -> float:
        """Return the objective value."""
        if self.result is None:
            msg = "The model is not optimized."
            raise Exception(msg)
        return self.result.objective_value()
