from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from ortools.math_opt.python import mathopt, parameters, result

from xplor.model import XplorModel
from xplor.types import VarType, cast_to_dtypes

if TYPE_CHECKING:
    from xplor.obj_expr import ExpressionRepr


class XplorMathOpt(XplorModel):
    """Xplor wrapper for the OR-Tools MathOpt solver.

    This class extends `XplorModel` to provide an interface for building
    and solving optimization problems using OR-Tools MathOpt.

    Attributes
    ----------
    model : mathopt.Model
        The underlying OR-Tools MathOpt model instance.
    vars : dict[str, pl.Series]
        A dictionary storing Polars Series of optimization variables,
        indexed by name.
    var_types : dict[str, VarType]
        A dictionary storing the `VarType` (CONTINUOUS, INTEGER, BINARY)
        for each variable series, indexed by its base name.
    result : result.SolveResult
        The result object returned by MathOpt after optimization.
        It contains solution status, objective value, and variable values.

    """

    model: mathopt.Model
    result: result.SolveResult

    def __init__(self, model: mathopt.Model | None = None) -> None:
        """Initialize the XplorMathOpt model wrapper.

        If no MathOpt model is provided, a new one is instantiated.

        Parameters
        ----------
        model : mathopt.Model | None, default None
            An optional, pre-existing MathOpt model instance.

        """
        model = mathopt.Model() if model is None else model
        super().__init__(model=model)

    def _add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: VarType = VarType.CONTINUOUS,
    ) -> pl.Series:
        """Return a series of MathOpt variables.

        Handles the conversion of Xplor's VarType to MathOpt's boolean `is_integer` flag.
        For "BINARY" types, bounds are explicitly clipped to [0, 1] as a prerequisite for MathOpt.

        Parameters
        ----------
        df : pl.DataFrame
            A DataFrame containing the columns ["lb", "ub", "obj", "name"].
        name : str
            The base name for the variables.
        vtype : VarType, default VarType.CONTINUOUS
            The type of the variable.

        Returns
        -------
        pl.Series
            A Polars Object Series containing the created MathOpt variable objects.

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

    def _add_constrs(self, df: pl.DataFrame, name: str, expr_repr: ExpressionRepr) -> pl.Series:
        """Return a series of MathOpt linear constraints.

        This method is called by `XplorModel.add_constrs` after the expression
        has been processed into rows of data and a constraint string.

        Parameters
        ----------
        df : pl.DataFrame
            A DataFrame containing the necessary components for the constraint expression.
        name : str
            The base name for the constraint.
        expr_repr : ExpressionRepr
            The evaluated string representation of the constraint expression.

        Returns
        -------
        pl.Series
            A Polars Object Series containing the created MathOpt constraint objects.

        """
        # TODO: manage non linear constraint
        # https://github.com/gab23r/xplor/issues/1

        return pl.Series(
            name,
            [
                self.model.add_linear_constraint(expr_repr.evaluate(row), name=f"{name}[{i}]")
                for i, row in enumerate(df.rows())
            ],
            dtype=pl.Object,
        )

    def optimize(self, solver_type: parameters.SolverType | None = None) -> None:  # ty:ignore[invalid-method-override]
        """Solve the MathOpt model.

        Uses `mathopt.solve()` to solve the model and stores the result internally.

        Parameters
        ----------
        solver_type : parameters.SolverType | None, default SolverType.GLOP
            The specific OR-Tools solver to use (e.g., GLOP, GSCIP).
            Defaults to MathOpt's native GLOP solver if none is provided.

        Examples
        --------
        1. Using the default solver (GLOP):
           >>> xmodel.optimize()

        2. Specifying a different solver (requires setup/licensing for commercial solvers):
           >>> from ortools.math_opt.python.parameters import SolverType
           >>> xmodel.optimize(solver_type=SolverType.GUROBI)

        """
        solver_type = mathopt.SolverType.GLOP if solver_type is None else solver_type
        self.result = mathopt.solve(self.model, solver_type)

    def get_objective_value(self) -> float:
        """Return the objective value from the solved MathOpt model.

        The value is read from the stored `result` object.

        Returns
        -------
        float
            The value of the objective function.

        Raises
        ------
        Exception
            If the model has not been optimized successfully (i.e., `self.result` is None).

        """
        if self.result is None:
            msg = "The model is not optimized."
            raise Exception(msg)
        return self.result.objective_value()

    def get_variable_values(self, name: str) -> pl.Series:
        """Read the optimal values of a variable series from the MathOpt solution.

        The method ensures the returned Polars Series has the correct data type
        (Float64 for continuous, Int64 for integer/binary).

        Parameters
        ----------
        name : str
            The base name used when the variable series was created with `xmodel.add_vars()`.

        Returns
        -------
        pl.Series
            A Polars Series containing the optimal variable values.

        Examples
        --------
        >>> # Assuming 'production' was the variable name used in add_vars
        >>> solution_series = xmodel.get_variable_values("production")
        >>> df_with_solution = df.with_columns(solution_series.alias("solution"))

        """
        return cast_to_dtypes(
            pl.Series(name, self.result.variable_values(self.vars[name])), self.var_types[name]
        )
