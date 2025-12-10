from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl

from xplor._utils import parse_into_expr, series_to_df
from xplor.types import VarType

if TYPE_CHECKING:
    from collections.abc import Sequence

    import gurobipy as gp
    from ortools.math_opt.python import mathopt

    from xplor.obj_expr import ExpressionRepr
    from xplor.var_expr import ConstrExpr


class XplorModel(ABC):
    """Abstract base class for all Xplor optimization model wrappers.

    Defines the core interface for adding variables and constraints
    to the underlying optimization model (e.g., MathOpt, Gurobi, etc.).

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

    def __init__(self, model: gp.Model | mathopt.Model) -> None:
        """Initialize the model wrapper.

        Parameters
        ----------
        model : gurobipy.Model | mathopt.Model
            The instantiated underlying solver model object.

        """
        self.model = model
        self.vars: dict[str, pl.Series] = {}
        self.var_types: dict[str, VarType] = {}

    def add_vars(
        self,
        name: str,
        *,
        lb: float | str | pl.Expr = 0.0,
        ub: float | str | pl.Expr | None = None,
        obj: float | str | pl.Expr = 0.0,
        indices: pl.Expr | None = None,
        vtype: VarType | None = None,
    ) -> pl.Expr:
        """Define and return a Var expression for optimization variables.

        This method generates a Polars expression that, when consumed (e.g., via
        `.with_columns()`), creates optimization variables for every row and adds
        them to the underlying solver model.

        Parameters
        ----------
        name : str
            The base name for the variables (e.g., "production" or "flow").
            This name is used to retrieve variable values after optimization.
        lb : float | str | pl.Expr, default 0.0
            Lower bound for created variables. Can be a scalar, a column name (str),
            or a Polars expression.
        ub : float | str | pl.Expr | None, default None
            Upper bound for created variables. If None, the solver default is used.
        obj: float | str | pl.Expr, default 0.0
            Objective function coefficient for created variables. Can be a scalar,
            a column name, or a Polars expression.
        indices: pl.Expr | None, default pl.row_index()
            Keys (column names) that uniquely identify each variable instance.
            Used to format the internal variable names (e.g., 'x[1,2]').
        vtype: VarType | None, default VarType.CONTINUOUS
            The type of the variable (CONTINUOUS, INTEGER, or BINARY).

        Returns
        -------
        pl.Expr
            A Polars expression (`Var`) that, when executed, adds variables to the model
            and returns them as an `Object` Series in the DataFrame.

        Examples
        --------
        Assuming `xmodel` is an instance of a concrete class (`XplorGurobi`).

        ```python
        # 1. Basic variable creation using columns for bounds:
        >>> data = pl.DataFrame({"max_limit": [10.0, 5.0]})
        >>> df = data.with_columns(
        ...     xmodel.add_vars("x", lb=0.0, ub=pl.col("max_limit"), obj=1.0)
        ... )
        # df["x"] now contains gurobipy.Var or mathopt.Variable objects.

        # s2. Creating integer variables indexed by two columns:
        >>> data = pl.DataFrame({"time": [1, 1, 2, 2], "resource": ["A", "B", "A", "B"]})
        >>> df = data.with_columns(
        ...     xmodel.add_vars(
        ...         "sched",
        ...         indices=["time", "resource"],
        ...         vtype=VarType.INTEGER,
        ...     )
        ... )
        # Variable names will look like 'sched[1,A]', 'sched[1,B]', etc.
        ```

        """
        indices = pl.row_index() if indices is None else indices
        vtype = VarType.CONTINUOUS if vtype is None else vtype
        return pl.map_batches(
            [
                parse_into_expr(lb).alias("lb"),
                parse_into_expr(ub).alias("ub"),
                parse_into_expr(obj).alias("obj"),
                pl.format(f"{name}[{{}}]", pl.concat_str(indices, separator=",")).alias("name"),
            ],
            lambda s: self._add_vars(series_to_df(s), name=name, vtype=vtype),
            return_dtype=pl.Object,
        ).alias(name)

    def add_constrs(
        self,
        *constr_exprs: ConstrExpr,
        indices: pl.Expr | None = None,
        **named_constr_exprs: ConstrExpr,
    ) -> pl.Expr:
        r"""Define and return a Constr expression for model constraints.

        This method accepts a symbolic relational expression (e.g., `x <= 5`)
        and generates a Polars expression that, when consumed (e.g., via `.select()`),
        adds the constraints to the underlying solver model.

        The constraint is added row-wise if the input expression is a Series of
        expressions, or as a single constraint if the expression is aggregated
        (e.g., using `.sum()`).

        Parameters
        ----------
        constr_exprs : ConstrExpr
            The constraints expression (e.g., a relational expression like
            `xplor.var("x").sum() <= 10`).
        indices: pl.Expr | None, default None
            Keys (column names) that uniquely identify each constraint instance.
            Used to format the internal variable names (e.g., 'constr[1,2]').
        named_constr_exprs : ConstrExpr
            Other constraints expression

        Returns
        -------
        pl.Expr
            A Polars expression (`Constr`) that, when executed, adds constraints
            to the model and returns them as an `Object` Series in the DataFrame.

        .. warning::
            All constraints provided within a single call to `add_constrs` should
            have the same granularity (i.e., correspond to the same set of indices).
            If constraints with different granularities are provided, Polars'
            broadcasting mechanism might lead to constraints being added multiple
            times to the optimization model.

        Examples
        --------
        Assuming `df` has been created and contains the variable Series `df["x"]`.

        ```python
        >>> df.select(
        ...     xmodel.add_constrs(
                    max_per_item = xplor.var("x") <= pl.col("capacity"),
                    min_per_item = xplor.var("x") >= pl.col("min_threshold"),
                    indices=["product"]
                ),
        ...     xmodel.add_constrs(total_min_prod = xplor.var("x").sum() >= 100.0)
        ... )

        ```

        """
        exprs: list[pl.Expr] = []
        constrs_repr_d: dict[str, ExpressionRepr] = {}
        for expr in constr_exprs:
            expr_repr, exprs = expr.parse(exprs)
            constrs_repr_d[expr._get_str(expr_repr, exprs)] = expr_repr
        constrs_repr_d |= {name: expr.parse(exprs)[0] for name, expr in named_constr_exprs.items()}
        indices = pl.lit(None, dtype=pl.Null) if indices is None else pl.struct(indices)

        return pl.map_batches(
            [*exprs, indices],
            lambda series: self._boardcast_and_add_constrs(series, constrs_repr_d),
            return_dtype=pl.Struct(dict.fromkeys(constrs_repr_d, pl.String)),
        ).struct.unnest()

    @abstractmethod
    def optimize(self, **kwargs: Any) -> Any:
        """Solve the model.

        This method triggers the underlying solver to find the optimal solution
        based on the defined variables, objective, and constraints.


        Returns
        -------
        Any
            The result object specific to the underlying solver (e.g., `result.SolveResult`
            for MathOpt, or None for Gurobi).

        """

    @abstractmethod
    def get_objective_value(self) -> float:
        """Return the objective value of the final solution.

        Returns
        -------
        float
            The value of the objective function from the solved model.

        Raises
        ------
        Exception
            If the model has not been optimized successfully.

        """

    @abstractmethod
    def get_variable_values(self, name: str) -> pl.Series:
        """Read the value of an optimization variable series from the solution.

        Parameters
        ----------
        name : str
            The base name used when the variable series was created with `xmodel.add_vars()`.

        Returns
        -------
        pl.Series
            A Polars Series (Float64 or Integer) containing the optimal values
            for the variables, aligned with the order of creation.

        """

    @abstractmethod
    def _add_constrs(
        self, df: pl.DataFrame, /, indices: pl.Series, **constrs_repr: ExpressionRepr
    ) -> None:
        """Return a series of constraints.

        This method iterates over the rows of the processed DataFrame to add constraints
        individually.

        Parameters
        ----------
        df : pl.DataFrame
            A DataFrame containing the necessary components for the constraint expression.
        indices: pl.Series
             A Series containing the indices for the constraint names.
        constrs_repr : ExpressionRepr
            The evaluated string representation of the constraint expression.

        """

    @abstractmethod
    def _add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: VarType = VarType.CONTINUOUS,
    ) -> pl.Series:
        """Return a series of variables.

        `df` should contains columns: ["lb", "ub", "obj, "name"].
        """

    def _boardcast_and_add_constrs(
        self, series: Sequence[pl.Series], constrs_repr_d: dict[str, ExpressionRepr]
    ) -> pl.Series:
        """Broadcast all series and call _add_constrs.

        The last series is a struct conaining the indices used for the constraint name.
        """
        # Convert series list back to a DataFrame
        df = series_to_df(series, rename=True)

        # Extract the last column (the index) as a Series
        indices = df[:, -1]
        if indices.dtype.base_type() is pl.Struct:
            indices = pl.select(pl.concat_str(df[:, -1].struct.unnest(), separator=",")).to_series()
        else:
            indices = df.select(pl.row_index().cast(pl.String)).to_series()

        self._add_constrs(
            df[:, 0:-1],
            **constrs_repr_d,
            indices=indices,
        )
        return (
            indices.to_frame("indices")
            # .select(**{name: pl.format(f"{name}[{{}}]", pl.col(indices)) for name in constrs_repr_d})
            .select(
                pl.struct(
                    [
                        pl.format(f"{name}[{{}}]", pl.col("indices")).alias(name)
                        for name in constrs_repr_d
                    ]
                )
            )
            .to_series()
        )
