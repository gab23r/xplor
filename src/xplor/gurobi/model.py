from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gurobipy as gp
import polars as pl

from xplor.model import XplorModel
from xplor.types import VarType, cast_to_dtypes

if TYPE_CHECKING:
    from xplor.obj_expr import ExpressionRepr


class XplorGurobi(XplorModel):
    """Xplor wrapper for the Gurobi solver.

    This class provides a specialized wrapper for Gurobi, translating XplorModel's
    abstract operations into Gurobi-specific API calls for defining variables,
    constraints, optimizing, and extracting solutions.

    Attributes
    ----------
    model : gurobipy.Model | mathopt.Model
        The instantiated underlying solver model object.
    vars : dict[str, pl.Series]
        A dictionary storing Polars Series of optimization variables,
        indexed by name.
    var_types : dict[str, VarType]
        A dictionary storing the `VarType` (CONTINUOUS, INTEGER, BINARY)
        for each variable series, indexed by its base name.

    """

    model: gp.Model

    def __init__(self, model: gp.Model | None = None) -> None:
        """Initialize the XplorGurobi model wrapper.

        If no Gurobi model is provided, a new one is instantiated.

        Parameters
        ----------
        model : gurobipy.Model | None, default None
            An optional, pre-existing Gurobi model instance.

        """
        model = gp.Model() if model is None else model
        super().__init__(model=model)

    def _add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: VarType = VarType.CONTINUOUS,
    ) -> pl.Series:
        """Return a series of gurobi variables`.

        This method leverages NumPy arrays from Polars columns to perform vectorized
        variable creation, setting the lower bound, upper bound, and objective
        coefficient in a single call.

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
            A Polars Object Series containing the created Gurobi variable objects.

        """
        self.var_types[name] = vtype
        self.vars[name] = pl.Series(
            self.model.addMVar(
                df.height,
                vtype=getattr(gp.GRB, vtype),
                lb=df["lb"].to_numpy(),
                ub=df["ub"].to_numpy(),
                obj=df["obj"].to_numpy(),
                name=df["name"].to_list(),
            ).tolist(),
            dtype=pl.Object(),
        )
        self.model.update()
        return self.vars[name]

    def _add_constrs(self, df: pl.DataFrame, name: str, expr_repr: ExpressionRepr) -> pl.Series:
        """Return a series of gurobi linear constraints.

        This method iterates over the rows of the processed DataFrame to add constraints
        individually, as Gurobi's `addConstrs` does not easily support complex vectorized
        Polars-derived expressions.

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
            A Polars Object Series containing the created Gurobi constraint objects.

        """
        # TODO: manage non linear constraint
        # https://github.com/gab23r/xplor/issues/1

        if df.height == 0:
            return pl.Series(name, dtype=pl.Object)

        # row = df.row(0)
        # lhs_constr_type = str(type(row[0]))
        # rhs_constr_type = str(type(row[1]))
        # if "GenExpr" in lhs_constr_type or "GenExpr" in rhs_constr_type:
        #     _add_constr = self.model.addConstr
        # elif "QuadExpr" in lhs_constr_type or "QuadExpr" in rhs_constr_type:
        #     _add_constr = self.model.addQConstr
        # else:
        #     _add_constr = self.model.addLConstr

        _add_constr = self.model.addLConstr
        series = pl.Series(
            [
                _add_constr(expr_repr.evaluate(row), name=f"{name}[{i}]")
                for i, row in enumerate(df.rows())
            ],
            dtype=pl.Object,
        )
        self.model.update()
        return series

    def optimize(self, **kwargs: Any) -> None:
        """Solve the Gurobi model.

        Calls the Gurobi model's built-in `optimize()` method. The `solver_type`
        parameter is accepted for API consistency with `XplorModel`, but is ignored,
        as Gurobi manages its own solver configuration.


        """
        self.model.optimize(**kwargs)

    def get_objective_value(self) -> float:
        """Return the objective value from the solved Gurobi model.

        The value is retrieved directly from the Gurobi model's objective object.

        Returns
        -------
        float
            The value of the objective function.

        """
        return self.model.getObjective().getValue()

    def get_variable_values(self, name: str) -> pl.Series:
        """Read the optimal values of a variable series from the Gurobi solution.

        The method reads the `.x` attribute from each Gurobi variable object and
        ensures the returned Polars Series has the correct data type
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
        >>> # Assuming 'flow' was the variable name used in add_vars
        >>> solution_series = xmodel.get_variable_values("flow")
        >>> df_with_solution = df.with_columns(solution_series.alias("solution"))

        """
        return cast_to_dtypes(pl.Series(name, [e.x for e in self.vars[name]]), self.var_types[name])
