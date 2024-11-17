from typing import Self

import gurobipy as gp
import polars as pl

import xplor.gurobi._utils as gurobi_utils
from xplor import _utils


class XplorGurobi:
    """Xplor base class to wrap your Gurobi model."""

    def __init__(
        self: Self, model: gp.Model, *, deterministic: bool = False, auto_update: bool = False
    ) -> None:
        """Initialize an XplorGurobi instance.

        Parameters
        ----------
        model : gp.Model
            The Gurobi optimization model to wrap.
        deterministic : bool, default=False
            Whether to ensure deterministic behavior when creating variables.
        auto_update : bool, default=False
            Whether to automatically update the model after adding variables or constraints.

        Notes
        -----
        The class maintains two dictionaries:
        - constrs: Stores constraints as DataFrames
        - vars: Stores variables as DataFrames

        """
        self.model = model
        self.deterministic = deterministic
        self.auto_update = auto_update
        self.constrs: dict[str, pl.DataFrame] = {}
        self.vars: dict[str, pl.DataFrame] = {}

    @gurobi_utils.update_model
    def add_constrs(
        self: Self,
        df: pl.DataFrame,
        expr: str,
        name: str | None = None,
        indices: list[str] | None = None,
    ) -> pl.DataFrame:
        """Add constraints for each row in the dataframe using a string expression.

        Parameters
        ----------
        df : pl.DataFrame
            The input DataFrame containing the data for creating constraints
        expr : str
            A string expression representing the constraint. Must include a comparison
            operator ('<=', '==', or '>='). The expression can reference column names
            and use standard mathematical operators. For example: "2*x + y <= z"
        name : str | None
            Base name for the constraints. If provided, constraints will be added as
            a new column to the DataFrame with this name.
        indices: list[str] | None
            Keys of the constraint

        Returns
        -------
        pl.DataFrame
            If name is provided, returns DataFrame with new constraints appended as a column.
            If name is None, returns the original DataFrame unchanged.

        Examples
        --------
        >>> df = pl.DataFrame({
        ...     "x": [gp.Var()],
        ...     "y": [gp.Var()],
        ...     "z": [5]
        ... })
        >>> df = df.pipe(xplor.add_constrs, model, "2*x + y <= z", name="capacity")

        Notes
        -----
        - Expression can use any column name from the DataFrame
        - Supports arithmetic operations (+, -, *, /) and Gurobi functions
        - Empty DataFrames are returned unchanged
        - The model is not automatically updated after adding constraints

        See Also
        --------
        add_vars : Function to add variables to the model

        """
        if df.height == 0:
            return df

        name = name or expr

        if indices is None:
            indices = ["__default_index__"]
            df = df.with_row_index(name=indices[0])

        df_ = df.sort(indices) if self.deterministic else df

        lhs_, sense_, rhs_ = _utils.evaluate_comp_expr(df_, expr)
        name_ = _utils.format_indices(df_, name, indices) if indices else name
        constrs = gurobi_utils.add_constrs_from_dataframe_args(
            df_, self.model, lhs_, sense_, rhs_, name_
        )

        # we need a join because of the sort
        # we take the advantage of checking that indices are unique on df
        # this is almost free via `validate`
        self.constrs[name] = df.join(
            df_.select(*indices, pl.Series(name, constrs, dtype=pl.Object)),
            on=indices,
            validate="1:1",
        )

        return self.constrs[name]

    @gurobi_utils.update_model
    def add_vars(
        self: Self,
        df: pl.DataFrame,
        name: str,
        vtype: str | None = None,
        *,
        lb: float | str | pl.Expr = 0.0,
        ub: float | str | pl.Expr | None = None,
        obj: float | str | pl.Expr = 0.0,
        indices: list[str] | None = None,
    ) -> pl.DataFrame:
        """Add a variable to the gurobi model for each row in the dataframe.

        Parameters
        ----------
        df: pl.DataFrame
            The dataframe that will hold the new variables
        name : str
            The variable name
        vtype: str
            The variable type for created variables
        lb : float | str | pl.Expr
            Lower bound for created variables.
        ub : float | str | pl.Expr
            Upper bound for created variables.
        obj: float | str | pl.Expr
            Objective function coefficient for created variables.
        indices: list[str] | None
            Keys of the variables


        Returns
        -------
        DataFrame
            A new DataFrame with new Vars appended as a column

        """
        if ub is None:
            ub = gp.GRB.INFINITY
        if vtype is None:
            vtype = gp.GRB.CONTINUOUS
        if indices is None:
            indices = ["__default_index__"]
            df = df.with_row_index(name=indices[0])

        df_ = df.select(*indices, lb=lb, ub=ub, obj=obj)
        if self.deterministic:
            df_ = df_.sort(indices)

        mvar = self.model.addMVar(
            df.height,
            vtype=vtype,
            lb=df_["lb"].to_numpy(),
            ub=df_["ub"].to_numpy(),
            obj=df_["obj"].to_numpy(),
            name=_utils.format_indices(df_, name, indices),
        ).tolist()

        # we need a join because of the sort
        # we take the advantage of checking that indices are unique on df
        # this is almost free via `validate`
        self.vars[name] = df.join(
            df_.select(*indices, pl.Series(name, mvar, dtype=pl.Object)), on=indices, validate="1:1"
        )

        return self.vars[name]
