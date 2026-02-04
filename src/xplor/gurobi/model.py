from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gurobipy as gp
import polars as pl

from xplor.model import XplorModel
from xplor.types import VarType, cast_to_dtypes

if TYPE_CHECKING:
    from gurobipy import TempLConstr


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
                lb=df["lb"].to_numpy() if df["lb"].dtype != pl.Null else None,
                ub=df["ub"].to_numpy() if df["ub"].dtype != pl.Null else None,
                obj=df["obj"].to_numpy() if df["obj"].dtype != pl.Null else None,
                name=df["name"].to_list(),
            ).tolist(),
            dtype=pl.Object(),
        )
        self.model.update()
        return self.vars[name]

    def _add_constr(self, tmp_constr: TempLConstr, name: str) -> None:
        self.model.addLConstr(tmp_constr, name=name)

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

    def read_values(self, name: pl.Expr) -> pl.Expr:
        """Read the value of an optimization variable.

        Parameters
        ----------
        name : pl.Expr
            Expression to evaluate.

        Returns
        -------
        pl.Expr
            Values of the variable expression.

        Examples
        --------
        >>> xmodel: XplorModel
        >>> df_with_solution = df.with_columns(xmodel.read_values(pl.selectors.object()))

        """

        def _extract(v: Any) -> float | None:
            if hasattr(v, "x"):
                return v.x
            if v is None:
                return None
            if hasattr(v, "getValue"):
                return v.getValue()
            return float(v)

        return name.map_batches(
            lambda d: cast_to_dtypes(
                pl.Series([_extract(v) for v in d]), self.var_types.get(d.name, VarType.CONTINUOUS)
            )
        )
