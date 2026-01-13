from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import polars as pl

from xplor._utils import parse_into_expr, series_to_df
from xplor.exprs.var import _ProxyVarExpr
from xplor.types import VarType

if TYPE_CHECKING:
    from xplor.exprs import ConstrExpr
    from xplor.exprs.obj import ExpressionRepr

# Type variable for expression terms (backend-specific)
ExpressionType = TypeVar("ExpressionType")

# Type variable for the underlying solver model (backend-specific)
ModelType = TypeVar("ModelType")


class XplorModel(ABC, Generic[ModelType, ExpressionType]):
    """Abstract base class for all Xplor optimization model wrappers.

    Defines the core interface for adding variables and constraints
    to the underlying optimization model (e.g., MathOpt, Gurobi, Hexaly).

    Type Parameters
    ----------------
    ObjTermsType
        The type of objective expression terms used by the backend
        (e.g., gp.LinExpr for Gurobi, LinearSum for MathOpt).
    ModelType
        The type of the underlying solver model object
        (e.g., gp.Model for Gurobi, mathopt.Model for MathOpt).

    Attributes
    ----------
    model : ModelType
        The instantiated underlying solver model object.
    vars : dict[str, pl.Series]
        A dictionary storing Polars Series of optimization variables,
        indexed by name.
    var_types : dict[str, VarType]
        A dictionary storing the `VarType` (CONTINUOUS, INTEGER, BINARY)
        for each variable series, indexed by its base name.

    """

    model: ModelType

    def __init__(self, model: ModelType) -> None:
        """Initialize the model wrapper.

        Parameters
        ----------
        model : ModelType
            The instantiated underlying solver model object.

        """
        self.model = model
        self.vars: dict[str, pl.Series] = {}
        self.var_types: dict[str, VarType] = {}
        self._priority_obj_terms: dict[int, ExpressionType] = {}

    @cached_property
    def var(self) -> _ProxyVarExpr:
        """The entry point for creating custom expression objects (VarExpr) that represent
        variables or columns used within a composite Polars expression chain.

        This proxy acts similarly to `polars.col()`, allowing you to reference
        optimization variables (created via `xmodel.add_vars()`) or standard DataFrame columns
        in a solver-compatible expression.

        The resulting expression object can be combined with standard Polars expressions
        to form constraints or objective function components.

        Examples:
        >>> xmodel = XplorMathOpt()
        >>> df = df.with_columns(xmodel.add_vars("production"))
        >>> df.select(total_cost = xmodel.var("production") * pl.col("cost"))

        """
        return _ProxyVarExpr()

    def add_vars(
        self,
        name: str,
        *,
        lb: float | str | pl.Expr = 0.0,
        ub: float | str | pl.Expr | None = None,
        obj: float | str | pl.Expr = 0.0,
        indices: pl.Expr | list[str] | None = None,
        vtype: VarType | None = None,
        priority: int = 0,
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
        priority: int, default 0
            Multi-objective optimization priority. Higher priority numbers are optimized
            FIRST (priority 2 before priority 1 before priority 0). All objectives with
            the same priority are combined into a single weighted sum. Currently only
            supported by the Gurobi backend.

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
        indices = pl.concat_str(pl.row_index() if indices is None else indices, separator=",")
        vtype = VarType.CONTINUOUS if vtype is None else vtype
        return pl.map_batches(
            [
                parse_into_expr(lb).alias("lb"),
                parse_into_expr(ub).alias("ub"),
                parse_into_expr(obj).alias("obj"),
                pl.format(f"{name}[{{}}]", indices).alias("name"),
            ],
            lambda s: self._add_vars(series_to_df(s), name=name, vtype=vtype, priority=priority),
            return_dtype=pl.Object,
        ).alias(name)

    def add_constrs(
        self,
        df: pl.DataFrame,
        *constr_exprs: ConstrExpr,
        indices: pl.Expr | list[str] | None = None,
        **named_constr_exprs: ConstrExpr,
    ) -> pl.DataFrame:
        r"""Define and return a Constr expression for model constraints.

        This method accepts a symbolic relational expression (e.g., `x <= 5`)
        and generates a Polars expression that, when consumed (e.g., via `.select()`),
        adds the constraints to the underlying solver model.

        The constraint is added row-wise if the input expression is a Series of
        expressions, or as a single constraint if the expression is aggregated
        (e.g., using `.sum()`).

        Parameters
        ----------
        df: pl.DataFrame
            The polars DataFrame used to create the constraints
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
        >>> df.pipe(
        ...     xmodel.add_constrs,
                max_per_item = xplor.var("x") <= pl.col("capacity"),
                min_per_item = xplor.var("x") >= pl.col("min_threshold"),
                indices=["product"]
        ... )

        ```

        """
        if isinstance(indices, list):
            indices = pl.concat_str(indices, separator=",")

        for expr in constr_exprs:
            name = str(expr)
            assert name not in named_constr_exprs, f"Duplicated name for constraint {name}"
            named_constr_exprs[name] = expr

        # Process multi-output and single-output constraints separately
        exprs: list[pl.Expr] = []
        constrs_repr_d: dict[str, ExpressionRepr] = {}

        for name, expr in named_constr_exprs.items():
            if expr.meta.has_multiple_outputs():
                self._process_multi_output_constraint(df, expr, indices)
            else:
                constrs_repr_d[name], exprs = expr.parse(exprs)
        if constrs_repr_d:
            self._process_single_output_constraints(df, constrs_repr_d, exprs, indices)

        return df

    def _process_multi_output_constraint(
        self, df: pl.DataFrame, expr: ConstrExpr, indices: pl.Expr | None
    ) -> None:
        """Process constraints with multiple outputs.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame used to create the constraints
        expr : ConstrExpr
            The constraint expression with multiple outputs
        indices : pl.Expr | None
            Optional indices for constraint naming

        """
        df_constrs = df.select(expr)
        for series in df_constrs.iter_columns():
            indices_series = self._get_indices_series(df, series, indices)
            for index, tmp_constr in zip(indices_series, series, strict=True):
                self._add_constr(tmp_constr, name=f"{series.name}[{index}]")

    def _process_single_output_constraints(
        self,
        df: pl.DataFrame,
        constrs_repr_d: dict[str, ExpressionRepr],
        exprs: list[pl.Expr],
        indices: pl.Expr | None,
    ) -> None:
        """Process constraints with single outputs.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame used to create the constraints
        constrs_repr_d : dict[str, ExpressionRepr]
            Dictionary mapping constraint names to their representations
        exprs : list[pl.Expr]
            List of Polars expressions to evaluate
        indices : pl.Expr | None
            Optional indices for constraint naming

        """
        df_constrs = df.select([expr.alias(str(i)) for i, expr in enumerate(exprs)])
        indices_series = (
            df_constrs.select(pl.row_index()) if indices is None else df.select(indices)
        ).to_series()
        self._add_constrs(df_constrs, **constrs_repr_d, indices=indices_series)

    def _get_indices_series(
        self, df: pl.DataFrame, series: pl.Series, indices: pl.Expr | None
    ) -> pl.Series:
        """Get the indices series for constraint naming.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame used to create the constraints
        series : pl.Series
            The constraint series
        indices : pl.Expr | None
            Optional indices expression

        Returns
        -------
        pl.Series
            The indices series for naming

        """
        if indices is None:
            return series.to_frame().select(pl.row_index()).to_series()
        return df.select(indices).to_series()

    @abstractmethod
    def _add_constr(self, tmp_constr: Any, name: str) -> None:
        pass

    def _add_constrs(
        self,
        df: pl.DataFrame,
        /,
        indices: pl.Series,
        **constrs_repr: ExpressionRepr,
    ) -> None:
        """Return a series of MathOpt linear constraints.

        This method is called by `XplorModel.add_constrs` after the expression
        has been processed into rows of data and a constraint string.

        Parameters
        ----------
        df : pl.DataFrame
            A DataFrame containing the necessary components for the constraint expression.
        indices: pl.Series
             A Series containing the indices for the constraint names.
        constrs_repr : ExpressionRepr
            The evaluated string representation of the constraint expression.

        """
        # TODO: manage non linear constraint
        # https://github.com/gab23r/xplor/issues/1

        for row, index in zip(df.rows(), indices, strict=True):
            for name, constr_repr in constrs_repr.items():
                self._add_constr(constr_repr.evaluate(row), name=f"{name}[{index}]")

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

    @abstractmethod
    def _add_vars(
        self,
        df: pl.DataFrame,
        name: str,
        vtype: VarType = VarType.CONTINUOUS,
        priority: int = 0,
    ) -> pl.Series:
        """Return a series of variables.

        `df` should contains columns: ["lb", "ub", "obj, "name"].
        `priority` indicates the optimization priority (higher values optimized first).
        """
