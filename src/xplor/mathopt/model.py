from __future__ import annotations

import polars as pl
from ortools.math_opt.python import mathopt, parameters, result

from xplor.model import XplorModel
from xplor.types import VarType, cast_to_dtypes


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
        name: str,
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
        self.var_types[name] = vtype
        self.vars[name] = pl.Series(
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
                sum([w * v for w, v in zip(df["obj"], self.vars[name], strict=True)])
            )

        return self.vars[name]

    def _add_constrs(self, df: pl.DataFrame, name: str, expr_str: str) -> pl.Series:
        """Return a series of variables."""
        # TODO: manage non linear constraint
        # https://github.com/gab23r/xplor/issues/1

        return pl.Series(
            [
                self.model.add_linear_constraint(eval(expr_str), name=f"{name}[{i}]")
                for i, d in enumerate(df.rows())
            ],
            dtype=pl.Object,
        )

    def optimize(self, solver_type: parameters.SolverType | None = None) -> None:
        """Solve the model."""
        solver_type = mathopt.SolverType.GLOP if solver_type is None else solver_type
        self.result = mathopt.solve(self.model, solver_type)

    def get_objective_value(self) -> float:
        """Return the objective value."""
        if self.result is None:
            msg = "The model is not optimized."
            raise Exception(msg)
        return self.result.objective_value()

    def get_variable_values(self, name: str) -> pl.Series:
        """Read the value of a variables."""
        return cast_to_dtypes(
            pl.Series(name, self.result.variable_values(self.vars[name])), self.var_types[name]
        )
