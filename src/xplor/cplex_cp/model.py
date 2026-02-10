"""CPLEX CP backend for constraint programming problems."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import polars as pl
from docplex.cp.model import (
    CpoExpr,
    CpoModel,
    CpoVariable,
    binary_var_list,
    float_var_list,
    integer_var_list,
    interval_var,
)

from xplor._utils import parse_into_expr, series_to_df
from xplor.cplex_cp.var import IntervalVarExpr, _ProxyCplexCPVarExpr
from xplor.model import XplorModel
from xplor.types import cast_to_dtypes

if TYPE_CHECKING:
    from docplex.cp.expression import CpoIntervalVar
    from docplex.cp.solution import CpoSolveResult


class XplorCplexCP(XplorModel[CpoModel, CpoVariable, CpoExpr]):
    """Xplor wrapper for CPLEX CP (Constraint Programming) solver.

    This class provides a Polars-based interface for constraint programming problems
    like scheduling (RCPSP, job shop) using CPLEX CP Optimizer via docplex.cp.

    Unlike the mathematical programming backends (XplorCplex, XplorGurobi), this
    uses interval variables and CP-specific constraints.

    Attributes
    ----------
    model : docplex.cp.model.CpoModel
        The underlying CPLEX CP model

    Examples
    --------
    >>> import polars as pl  # doctest: +SKIP
    >>> from xplor.cplex_cp import XplorCplexCP  # doctest: +SKIP
    >>> xmodel = XplorCplexCP()  # doctest: +SKIP
    >>> # Create tasks DataFrame
    >>> df = pl.DataFrame({  # doctest: +SKIP
    ...     "task": ["T1", "T2", "T3"],
    ...     "duration": [3, 5, 4]
    ... })
    >>> # Add interval variables
    >>> df = df.with_columns(xmodel.add_interval_vars("iv", duration="duration"))  # doctest: +SKIP
    >>> # Add precedence constraint: T1 must finish before T2 starts
    >>> xmodel.add_constr(  # doctest: +SKIP
    ...     xmodel.var.iv.filter(pl.col("task") == "T1")
    ...         .end_before_start(xmodel.var.iv.filter(pl.col("task") == "T2"))
    ... )

    """

    def __init__(self, model: CpoModel | None = None) -> None:
        """Initialize the XplorCplexCP wrapper.

        Parameters
        ----------
        model : docplex.cp.model.CpoModel | None, default None
            Optional pre-existing CPLEX CP model. If None, creates a new model.

        """
        self.solution: CpoSolveResult | None = None
        super().__init__(CpoModel() if model is None else model)

    @cached_property
    def var(self) -> _ProxyCplexCPVarExpr:
        """Entry point for creating interval variable expressions.

        Similar to polars.col(), allows referencing interval variables created
        via add_interval_vars() in constraint expressions.

        Returns
        -------
        _ProxyCplexCPVarExpr
            Proxy object for creating interval variable expressions

        Examples
        --------
        >>> xmodel.var("task_iv")  # doctest: +SKIP
        >>> xmodel.var.task_iv  # doctest: +SKIP

        """
        return _ProxyCplexCPVarExpr()

    def read_values(self, name: pl.Expr) -> pl.Expr:
        """Read the value of an optimization variable from the solution.

        For interval variables, returns a struct with start, end, length, and present fields.
        For regular variables, returns the variable value.

        Parameters
        ----------
        name : pl.Expr
            Expression to evaluate.

        Returns
        -------
        pl.Expr
            Values of the variable expression. For interval variables, returns a struct
            with fields: start (Int64), end (Int64), length (Int64), present (Boolean).

        Examples
        --------
        >>> xmodel: XplorModel
        >>> # For regular variables
        >>> df_with_solution = df.with_columns(xmodel.read_values(pl.col("x")))  # doctest: +SKIP
        >>> # For interval variables - returns struct with start/end/length/present
        >>> df = df.with_columns(xmodel.read_values(pl.col("task_iv")))  # doctest: +SKIP
        >>> df = df.with_columns(  # doctest: +SKIP
        ...     start=pl.col("task_iv").struct.field("start"),
        ...     end=pl.col("task_iv").struct.field("end"),
        ... )

        """

        def _extract_interval(v: Any) -> dict[str, Any]:
            """Extract interval variable solution as struct."""
            if v is None or self.solution is None:
                return {"start": None, "end": None, "length": None, "present": None}
            sol = self.solution.get_var_solution(v)
            return {
                "start": sol.start,
                "end": sol.end,
                "length": sol.length,
                "present": sol.is_present(),
            }

        def _extract_regular(v: Any) -> float | None:
            """Extract regular variable solution value."""
            if v is None:
                return None
            if hasattr(v, "solution_value"):
                return v.solution_value
            return float(v)

        def _batch_extract(series: pl.Series) -> pl.Series:
            """Determine variable type and extract appropriately."""
            if len(series) == 0:
                return series

            # Check if this is an interval variable column (internal marker)
            if self.var_types.get(series.name) == "_INTERVAL_CP":
                # Return struct with start/end/length/present
                data = [_extract_interval(v) for v in series]
                return pl.Series(
                    series.name,
                    data,
                    dtype=pl.Struct(
                        {
                            "start": pl.Int64,
                            "end": pl.Int64,
                            "length": pl.Int64,
                            "present": pl.Boolean,
                        }
                    ),
                )
            else:
                # Regular variable - return scalar value
                return cast_to_dtypes(
                    pl.Series([_extract_regular(v) for v in series]),
                    self.var_types.get(series.name, "CONTINUOUS"),
                )

        return name.map_batches(_batch_extract)

    def _add_continuous_vars(
        self,
        names: list[str],
        lb: list[float] | None,
        ub: list[float] | None,
    ) -> list[CpoVariable]:
        return float_var_list(len(names), lb, ub, names)

    def _add_integer_vars(
        self,
        names: list[str],
        lb: list[float] | None,
        ub: list[float] | None,
    ) -> list[CpoVariable]:
        return integer_var_list(len(names), lb, ub, names)

    def _add_binary_vars(
        self,
        names: list[str],
    ) -> list[CpoVariable]:
        return binary_var_list(len(names), name=names)

    def add_interval_vars(
        self,
        name: str,
        *,
        start: float | tuple[int, int] | pl.Expr | None = None,
        end: float | tuple[int, int] | pl.Expr | None = None,
        duration: float | tuple[int, int] | pl.Expr | None = None,
        length: float | tuple[int, int] | pl.Expr | None = None,
        optional: bool | pl.Expr = False,
    ) -> pl.Expr:
        """Create interval variables for each row in the DataFrame.

        Interval variables represent tasks/activities with start time, end time,
        and duration. They can be optional (present/absent).

        Parameters
        ----------
        name : str
            Name for the interval variable column
        start : int | float | tuple[int, int] | pl.Expr | None, default None
            Start time or (min, max) bounds. If None, unbounded.
        end : int | float | tuple[int, int] | pl.Expr | None, default None
            End time or (min, max) bounds. If None, unbounded.
        duration : int | float | tuple[int, int] | pl.Expr | None, default None
            Duration or (min, max) bounds. If None, computed from start/end.
        length : int | float | tuple[int, int] | pl.Expr | None, default None
            Length (actual work time, may differ from duration). If None, equals duration.
        optional : bool | pl.Expr, default False
            Whether intervals can be absent (not scheduled)

        Returns
        -------
        pl.Expr
            Polars expression that creates interval variables when materialized

        Examples
        --------
        >>> df = df.with_columns(  # doctest: +SKIP
        ...     xmodel.add_interval_vars("task", duration=pl.col("task_duration"))
        ... )

        """
        return pl.map_batches(
            [
                parse_into_expr(start).alias("start"),  # ty:ignore[invalid-argument-type]
                parse_into_expr(end).alias("end"),  # ty:ignore[invalid-argument-type]
                parse_into_expr(duration).alias("duration"),  # ty:ignore[invalid-argument-type]
                parse_into_expr(length).alias("length"),  # ty:ignore[invalid-argument-type]
                parse_into_expr(optional).alias("optional"),
                pl.format(f"{name}[{{}}]", pl.row_index()).alias("var_name"),
            ],
            lambda s: self._add_interval_vars_wrapper(series_to_df(s), name),
            return_dtype=pl.Object,
        ).alias(name)

    def _add_interval_vars_wrapper(self, df: pl.DataFrame, name: str) -> pl.Series:
        """Create interval variables from a DataFrame of parameters.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with columns: start, end, duration, length, optional, var_name
        name : str
            Name for the interval variable series

        Returns
        -------
        pl.Series
            Series of interval variables

        """
        # Create interval variables
        intervals = [
            interval_var(
                start=s,
                end=e,
                size=d,
                length=len_val,
                optional=o,
                name=n,
            )
            for s, e, d, len_val, o, n in df.with_columns(pl.col("optional").fill_null(False))
            .select("start", "end", "duration", "length", "optional", "var_name")
            .iter_rows()
        ]

        # Store interval variables
        result = pl.Series(name, intervals, dtype=pl.Object)
        self.vars[name] = result
        # Use internal marker for interval variables (not part of public VariableType)
        self.var_types[name] = "_INTERVAL_CP"  # type: ignore
        return result

    def _add_constr(self, tmp_constr: Any, name: str) -> None:
        self.model.add(tmp_constr)

    def _materialize_interval_expr(self, expr: IntervalVarExpr | pl.Expr) -> list[CpoIntervalVar]:
        """Convert interval expression to list of actual interval variables.

        Parameters
        ----------
        expr : IntervalVarExpr | pl.Expr
            The expression to materialize

        Returns
        -------
        list[CpoIntervalVar]
            List of CPLEX CP interval variables

        """
        if isinstance(expr, IntervalVarExpr):
            # Get the column name from the expression
            # This is a simplified approach - may need refinement
            col_name = str(expr._name).replace("col(", "").replace(")", "").strip('"').strip("'")
            if col_name in self.vars:
                return self.vars[col_name].to_list()
            msg = f"Interval variable column '{col_name}' not found"
            raise ValueError(msg)
        elif isinstance(expr, pl.Expr):
            # Try to extract column name
            col_name = str(expr).replace("col(", "").replace(")", "").strip('"').strip("'")
            if col_name in self.vars:
                return self.vars[col_name].to_list()
            msg = f"Interval variable column '{col_name}' not found"
            raise ValueError(msg)
        else:
            msg = f"Invalid expression type: {type(expr)}"
            raise TypeError(msg)

    def minimize_makespan(self, intervals: pl.Expr | str) -> None:
        """Set objective to minimize makespan (maximum end time).

        Parameters
        ----------
        intervals : pl.Expr | str
            The interval variables column name or expression

        Examples
        --------
        >>> xmodel.minimize_makespan("task_iv")  # doctest: +SKIP

        """
        if isinstance(intervals, str):
            intervals = pl.col(intervals)

        ivars = self._materialize_interval_expr(intervals)
        # Minimize the maximum end time
        self.model.add(self.model.minimize(self.model.max([self.model.end_of(iv) for iv in ivars])))  # ty:ignore[unresolved-attribute]

    def optimize(self, **kwargs: Any) -> None:
        """Solve the CP model.

        Parameters
        ----------
        **kwargs : Any
            Additional parameters passed to model.solve()
            Common parameters:
            - TimeLimit: Maximum solve time in seconds
            - LogVerbosity: Verbosity level for solver output

        """
        self.solution = self.model.solve(**kwargs)

    def get_objective_value(self) -> float:
        """Return the objective value from the solved model.

        Returns
        -------
        float
            The objective value

        Raises
        ------
        ValueError
            If the model has not been solved

        """
        if self.solution is None:
            msg = "Model has not been optimized"
            raise ValueError(msg)
        return self.solution.get_objective_values()[0]
