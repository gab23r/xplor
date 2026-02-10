"""Interval variable expressions for CPLEX CP backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

import polars as pl
from docplex.cp.modeler import (
    abs as cp_abs,
)
from docplex.cp.modeler import (
    ceil,
    end_at_end,
    end_before_end,
    end_before_start,
    end_of,
    exponent,
    floor,
    length_of,
    log,
    no_overlap,
    power,
    presence_of,
    sgn,
    size_of,
    square,
    start_at_end,
    start_at_start,
    start_before_end,
    start_before_start,
    start_of,
    trunc,
)
from docplex.cp.modeler import (
    max as cp_max,
)
from docplex.cp.modeler import (
    min as cp_min,
)
from docplex.cp.modeler import (
    round as cp_round,
)
from docplex.cp.modeler import (
    sum as cp_sum,
)

from xplor._utils import series_to_df
from xplor.exprs import ConstrExpr, VarExpr

if TYPE_CHECKING:
    from docplex.cp.model import CpoModel


class IntervalVarExpr(VarExpr):
    """Expression wrapper for CPLEX CP interval variables.

    Provides methods for creating CP constraints on interval variables such as
    precedence, synchronization, and resource constraints.

    Examples
    --------
    >>> from xplor.cplex_cp import XplorCplexCP  # doctest: +SKIP
    >>> xmodel = XplorCplexCP()  # doctest: +SKIP
    >>> # Create precedence constraint
    >>> xmodel.add_constr(xmodel.var.task1.end_before_start(xmodel.var.task2))  # doctest: +SKIP

    """

    _model: CpoModel

    def _create_binary_operation(
        self,
        other: VarExpr | pl.Expr | float,
        operation_func: Callable[[Any, Any], Any],
    ) -> VarExpr:
        """Create binary mathematical operation on interval expressions.

        Parameters
        ----------
        other : VarExpr | pl.Expr | float
            The second operand
        operation_func : Callable
            The docplex.cp.modeler function to apply (e.g., power, min, max)

        Returns
        -------
        IntervalVarExpr
            Expression representing the operation result

        """
        if not isinstance(other, pl.Expr):
            other = pl.lit(other)

        def func(df: pl.DataFrame) -> pl.Series:
            return pl.Series(
                [operation_func(row[0], row[1]) for row in df.rows()],
                dtype=pl.Object,
            )

        return IntervalVarExpr(
            pl.map_batches(
                [self, other],
                lambda series: func(series_to_df(series)),
                return_dtype=pl.Object,
            )
        )

    def _create_pairwise_constraint(
        self,
        other: IntervalVarExpr | pl.Expr,
        delay: pl.Expr | float,
        constraint_func: Callable[[Any, Any, float], Any],
    ) -> ConstrExpr:
        """Create pairwise interval constraint using docplex.cp.modeler function.

        Parameters
        ----------
        other : IntervalVarExpr | pl.Expr
            The other interval variable
        delay : pl.Expr | float
            Delay parameter
        constraint_func : Callable
            The docplex.cp.modeler constraint function to apply

        Returns
        -------
        ConstrExpr
            Constraint expression

        """
        if not isinstance(delay, pl.Expr):
            delay = pl.lit(delay)

        def func(df: pl.DataFrame) -> pl.Series:
            return pl.Series(
                [constraint_func(row[0], row[1], row[2]) for row in df.rows()],
                dtype=pl.Object,
            )

        return ConstrExpr(
            pl.map_batches(
                [self, other, delay],
                lambda series: func(series_to_df(series)),
                return_dtype=pl.Object,
            )
        )

    def end_before_start(
        self, other: IntervalVarExpr | pl.Expr, delay: pl.Expr | float = 0
    ) -> ConstrExpr:
        """Create precedence constraint: this interval must end before other starts.

        Parameters
        ----------
        other : IntervalVarExpr | pl.Expr
            The interval that should start after this one ends
        delay : int | float | pl.Expr, default 0
            Minimum delay between end of this and start of other

        Returns
        -------
        ConstrExpr
            Constraint expression

        Examples
        --------
        >>> xmodel.add_constrs(df, precedence=var.task1.end_before_start(var.task2, delay=2))  # doctest: +SKIP

        """
        return self._create_pairwise_constraint(other, delay, end_before_start)

    def start_at_start(
        self, other: IntervalVarExpr | pl.Expr, delay: pl.Expr | float = 0
    ) -> ConstrExpr:
        """Synchronize interval starts with optional delay.

        Parameters
        ----------
        other : IntervalVarExpr | pl.Expr
            The interval to synchronize with
        delay : int | float | pl.Expr, default 0
            Delay between starts

        Returns
        -------
        ConstrExpr
            Constraint expression

        """
        return self._create_pairwise_constraint(other, delay, start_at_start)

    def end_at_end(
        self, other: IntervalVarExpr | pl.Expr, delay: pl.Expr | float = 0
    ) -> ConstrExpr:
        """Synchronize interval ends with optional delay.

        Parameters
        ----------
        other : IntervalVarExpr | pl.Expr
            The interval to synchronize with
        delay : int | float | pl.Expr, default 0
            Delay between ends

        Returns
        -------
        ConstrExpr
            Constraint expression

        """
        return self._create_pairwise_constraint(other, delay, end_at_end)

    def start_at_end(
        self, other: IntervalVarExpr | pl.Expr, delay: pl.Expr | float = 0
    ) -> ConstrExpr:
        """Constrain this interval to start when other ends.

        Parameters
        ----------
        other : IntervalVarExpr | pl.Expr
            The interval that should end when this starts
        delay : int | float | pl.Expr, default 0
            Delay between end of other and start of this

        Returns
        -------
        ConstrExpr
            Constraint expression

        """
        return self._create_pairwise_constraint(other, delay, start_at_end)

    def start_before_end(
        self, other: IntervalVarExpr | pl.Expr, delay: pl.Expr | float = 0
    ) -> ConstrExpr:
        """Constrain this interval to start before other ends.

        Parameters
        ----------
        other : IntervalVarExpr | pl.Expr
            The interval that should end after this starts
        delay : int | float | pl.Expr, default 0
            Minimum delay

        Returns
        -------
        ConstrExpr
            Constraint expression

        """
        return self._create_pairwise_constraint(other, delay, start_before_end)

    def start_before_start(
        self, other: IntervalVarExpr | pl.Expr, delay: pl.Expr | float = 0
    ) -> ConstrExpr:
        """Constrain this interval to start before other starts.

        Parameters
        ----------
        other : IntervalVarExpr | pl.Expr
            The interval that should start after this starts
        delay : int | float | pl.Expr, default 0
            Minimum delay

        Returns
        -------
        ConstrExpr
            Constraint expression

        """
        return self._create_pairwise_constraint(other, delay, start_before_start)

    def end_before_end(
        self, other: IntervalVarExpr | pl.Expr, delay: pl.Expr | float = 0
    ) -> ConstrExpr:
        """Constrain this interval to end before other ends.

        Parameters
        ----------
        other : IntervalVarExpr | pl.Expr
            The interval that should end after this ends
        delay : int | float | pl.Expr, default 0
            Minimum delay

        Returns
        -------
        ConstrExpr
            Constraint expression

        """
        return self._create_pairwise_constraint(other, delay, end_before_end)

    def no_overlap(self) -> ConstrExpr:
        """Create no-overlap constraint for intervals in this column.

        Ensures that intervals do not overlap in time (disjunctive constraint).
        This creates a single constraint that applies to all intervals in the column.

        Returns
        -------
        VarExpr
            Constraint expression containing the no_overlap constraint

        Examples
        --------
        >>> xmodel.add_constrs(df, machine1_no_overlap=var.task_iv.no_overlap())  # doctest: +SKIP

        """

        def func(series: pl.Series) -> pl.Series:
            # Get all interval variables from the column
            intervals = series.to_list()
            # Create a single no_overlap constraint
            constraint = no_overlap(intervals)
            # Return as a single-element series
            return pl.Series([constraint], dtype=pl.Object)

        return ConstrExpr(self.map_batches(func, return_dtype=pl.Object))

    def start_of(self) -> VarExpr:
        """Get the start time of interval variables.

        Returns
        -------
        VarExpr
            Expression representing the start time of each interval

        Examples
        --------
        >>> xmodel.add_constrs(df, start_constraint=var.task_iv.start_of() >= 10)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(start_of, return_dtype=pl.Object))

    def end_of(self) -> IntervalVarExpr:
        """Get the end time of interval variables.

        Returns
        -------
        IntervalVarExpr
            Expression representing the end time of each interval

        Examples
        --------
        >>> xmodel.add_constrs(df, end_constraint=var.task_iv.end_of() <= 100)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(end_of, return_dtype=pl.Object))

    def length_of(self) -> IntervalVarExpr:
        """Get the length of interval variables.

        Length is the actual work time (may differ from size/duration).

        Returns
        -------
        IntervalVarExpr
            Expression representing the length of each interval

        Examples
        --------
        >>> xmodel.add_constrs(df, length_constraint=var.task_iv.length_of() == 5)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(length_of, return_dtype=pl.Object))

    def size_of(self) -> IntervalVarExpr:
        """Get the size (duration) of interval variables.

        Size is the duration from start to end.

        Returns
        -------
        IntervalVarExpr
            Expression representing the size of each interval

        Examples
        --------
        >>> xmodel.add_constrs(df, size_constraint=var.task_iv.size_of() >= pl.col("min_duration"))  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(size_of, return_dtype=pl.Object))

    def presence_of(self) -> IntervalVarExpr:
        """Get the presence status of optional interval variables.

        Returns
        -------
        IntervalVarExpr
            Expression representing whether each interval is present (scheduled)

        Examples
        --------
        >>> xmodel.add_constrs(df, presence_constraint=var.task_iv.presence_of() == 1)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(presence_of, return_dtype=pl.Object))

    # Mathematical operations - Unary
    def abs(self) -> IntervalVarExpr:
        """Absolute value of the expression.

        Returns
        -------
        IntervalVarExpr
            Expression representing the absolute value

        Examples
        --------
        >>> xmodel.add_constrs(df, abs_constraint=var.task_iv.start_of().abs() <= 100)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(cp_abs, return_dtype=pl.Object))

    def square(self) -> IntervalVarExpr:
        """Square of the expression.

        Returns
        -------
        IntervalVarExpr
            Expression representing the square

        Examples
        --------
        >>> xmodel.add_constrs(df, square_constraint=var.task_iv.start_of().square() <= 100)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(square, return_dtype=pl.Object))

    def cp_log(self) -> IntervalVarExpr:
        """Natural logarithm of the expression (CP version).

        Returns
        -------
        IntervalVarExpr
            Expression representing the natural logarithm

        Examples
        --------
        >>> xmodel.add_constrs(df, log_constraint=var.task_iv.start_of().cp_log() >= 1)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(log, return_dtype=pl.Object))

    def exponent(self) -> IntervalVarExpr:
        """Exponential (e^x) of the expression.

        Returns
        -------
        IntervalVarExpr
            Expression representing e^x

        Examples
        --------
        >>> xmodel.add_constrs(df, exp_constraint=var.task_iv.start_of().exponent() <= 1000)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(exponent, return_dtype=pl.Object))

    def ceil(self) -> IntervalVarExpr:
        """Ceiling of the expression.

        Returns
        -------
        IntervalVarExpr
            Expression representing the ceiling

        Examples
        --------
        >>> xmodel.add_constrs(df, ceil_constraint=var.task_iv.start_of().ceil() == pl.col("rounded_start"))  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(ceil, return_dtype=pl.Object))

    def floor(self) -> IntervalVarExpr:
        """Floor of the expression.

        Returns
        -------
        IntervalVarExpr
            Expression representing the floor

        Examples
        --------
        >>> xmodel.add_constrs(df, floor_constraint=var.task_iv.start_of().floor() >= 0)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(floor, return_dtype=pl.Object))

    def cp_round(self) -> IntervalVarExpr:
        """Round to nearest integer (CP version).

        Returns
        -------
        IntervalVarExpr
            Expression representing the rounded value

        Examples
        --------
        >>> xmodel.add_constrs(df, round_constraint=var.task_iv.start_of().cp_round() == pl.col("start"))  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(cp_round, return_dtype=pl.Object))

    def trunc(self) -> IntervalVarExpr:
        """Truncate to integer (remove fractional part).

        Returns
        -------
        IntervalVarExpr
            Expression representing the truncated value

        Examples
        --------
        >>> xmodel.add_constrs(df, trunc_constraint=var.task_iv.start_of().trunc() >= 0)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(trunc, return_dtype=pl.Object))

    def sgn(self) -> IntervalVarExpr:
        """Sign of the expression (-1, 0, or 1).

        Returns
        -------
        IntervalVarExpr
            Expression representing the sign

        Examples
        --------
        >>> xmodel.add_constrs(df, sign_constraint=var.task_iv.start_of().sgn() >= 0)  # doctest: +SKIP

        """
        return IntervalVarExpr(self.map_elements(sgn, return_dtype=pl.Object))

    # Mathematical operations - Binary
    def power(self, exponent: VarExpr | pl.Expr | float) -> VarExpr:
        """Raise expression to a power.

        Parameters
        ----------
        exponent : VarExpr | pl.Expr | float
            The exponent

        Returns
        -------
        IntervalVarExpr
            Expression representing the power

        Examples
        --------
        >>> xmodel.add_constrs(df, power_constraint=var.task_iv.start_of().power(2) <= 100)  # doctest: +SKIP

        """
        return self._create_binary_operation(exponent, power)

    def minimum(self, other: VarExpr | pl.Expr | float) -> VarExpr:
        """Element-wise minimum of this expression and another value.

        Parameters
        ----------
        other : VarExpr | pl.Expr | float
            The other value to compare

        Returns
        -------
        IntervalVarExpr
            Expression representing the element-wise minimum

        Examples
        --------
        >>> xmodel.add_constrs(df, min_constraint=var.task_iv.start_of().minimum(10) >= 0)  # doctest: +SKIP

        """
        return self._create_binary_operation(other, cp_min)

    def maximum(self, other: VarExpr | pl.Expr | float) -> VarExpr:
        """Element-wise maximum of this expression and another value.

        Parameters
        ----------
        other : VarExpr | pl.Expr | float
            The other value to compare

        Returns
        -------
        VarExpr
            Expression representing the element-wise maximum

        Examples
        --------
        >>> xmodel.add_constrs(df, max_constraint=var.task_iv.start_of().maximum(0) <= 100)  # doctest: +SKIP

        """
        return self._create_binary_operation(other, cp_max)

    # Aggregation operations
    def sum(self) -> VarExpr:
        """Sum all values in the expression.

        Returns
        -------
        VarExpr
            Expression representing the sum (single value)

        Examples
        --------
        >>> xmodel.add_constrs(df, sum_constraint=var.task_iv.start_of().sum() <= 1000)  # doctest: +SKIP

        """

        def func(series: pl.Series) -> pl.Series:
            # Sum aggregates all values into a single result
            total = cp_sum(series.to_list())
            return pl.Series([total], dtype=pl.Object)

        return VarExpr(self.map_batches(func, return_dtype=pl.Object))


class _ProxyCplexCPVarExpr:
    """Proxy class for creating CPLEX CP interval variable expressions."""

    def __call__(self, name: str, /, *more_names: str) -> IntervalVarExpr:
        """Create an interval variable expression from column name(s).

        Parameters
        ----------
        name : str
            Primary column name
        *more_names : str
            Additional column names

        Returns
        -------
        IntervalVarExpr
            Expression wrapping the specified columns

        """
        return IntervalVarExpr(pl.col(name, *more_names))

    def __getattr__(self, name: str) -> IntervalVarExpr:
        """Create an interval variable expression via attribute access.

        Parameters
        ----------
        name : str
            Column name

        Returns
        -------
        IntervalVarExpr
            Expression wrapping the specified column

        """
        return IntervalVarExpr(pl.col(name))
