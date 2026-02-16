from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import gurobipy as gp
import gurobipy.nlfunc
import polars as pl

from xplor.exprs import VarExpr
from xplor.gurobi.utils import mlinexpr_to_linexpr_list

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def first_gurobi_expr(
    s: pl.Series,
) -> gp.Var | gp.LinExpr | gp.NLExpr | gp.QuadExpr | gp.GenExpr | None:
    """Find the first Gurobi expression object in a Series.

    Loops through series values to find the first Gurobi object, which is more
    efficient than using Series.first(ignore_nulls=True) for Object dtype series.

    Parameters
    ----------
    s : pl.Series
        Series potentially containing Gurobi expression objects.

    Returns
    -------
    gp.Var | gp.LinExpr | gp.NLExpr | gp.QuadExpr | gp.GenExpr | None
        First Gurobi expression found, or None if no Gurobi objects exist.

    """
    for val in s:
        if isinstance(val, (gp.Var, gp.LinExpr, gp.NLExpr, gp.QuadExpr, gp.GenExpr)):
            return val
    return None


def to_mvar_or_mlinexpr(s: pl.Series) -> gp.MVar | gp.MLinExpr | np.ndarray:
    """Convert a Polars Series to Gurobi MVar, MLinExpr, or NumPy array.

    Scans the entire series to determine the appropriate Gurobi type:
    - If ANY element is GenExpr/QuadExpr/NLExpr → numpy array (row-by-row processing)
    - Else if ANY element is LinExpr → MLinExpr (vectorized)
    - Else if ANY element is Var → MVar (fastest vectorized)
    - Else → numpy array (numeric values)

    Parameters
    ----------
    s : pl.Series
        Series containing Gurobi variables, linear expressions, or numeric values.

    Returns
    -------
    gp.MVar | gp.MLinExpr | np.ndarray
        Vectorized Gurobi array or NumPy array.

    """
    if s.dtype == pl.Object:
        first: Any = s.first(ignore_nulls=True)
        if isinstance(first, gp.Var):
            if s.null_count() == 0:
                return gp.MVar(s)  # ty:ignore[too-many-positional-arguments]
            return gp.MLinExpr._from_linexprs(s.to_numpy())  # ty:ignore[unresolved-attribute]
        if isinstance(first, gp.LinExpr):
            return gp.MLinExpr(s.to_numpy())  # ty:ignore[too-many-positional-arguments]

    return s.to_numpy()


class GurobiVarExpr(VarExpr):
    """Gurobi-specific VarExpr with optimized vectorized operations.

    Extends VarExpr with Gurobi-specific optimizations using MVar/MLinExpr
    for efficient vectorized evaluation.

    Methods
    -------
        sum(): Vectorized sum using MVar/MLinExpr.
        any(): Creates a Gurobi OR constraint across elements in each group.
        all(): Creates a Gurobi AND constraint across elements in each group.
        abs(): Applies Gurobi's absolute value function.
        max(): Returns the maximum of elements in each group.
        min(): Returns the minimum of elements in each group.
        square(): Creates quadratic expressions (x^2).
        exp(): Exponential function (e^x).
        log(): Natural logarithm (ln(x)).
        sin(): Sine function.
        cos(): Cosine function.
        sqrt(): Square root.

    """

    def _to_expr(self) -> pl.Expr:
        if not self._nodes:
            return self._expr
        expr_repr, exprs = self.parse()

        def vectorized_eval(series: Sequence[pl.Series], **kwargs: Any) -> pl.Series:
            """Evaluate expression using vectorized NumPy operations."""
            gp_obj = tuple(map(to_mvar_or_mlinexpr, series))
            result = expr_repr.evaluate(gp_obj)

            # MLinExpr needs to be dispatched into list of LinExpr
            if isinstance(result, gp.MLinExpr):
                result = mlinexpr_to_linexpr_list(result)
            return pl.Series(result, dtype=pl.Object)

        return pl.map_batches(
            exprs,
            vectorized_eval,
            return_dtype=pl.Object,
        )

    def sum(
        self,
    ) -> Self:
        """Vectorized sum using Gurobi MVar/MLinExpr for optimal performance.

        Optimized pattern: (var * coeff).sum() uses gp.LinExpr(coeffs, vars)
        directly, which is much faster than element-wise multiplication + sum.

        Examples
        --------
        >>> df.group_by('group').agg(xmodel.var("x").sum())
        >>> df.select(total=(var.x + var.y).sum())  # Vectorized!
        >>> df.select(total=(var.x * pl.col("cost")).sum())  # Optimized LinExpr!

        """
        name = str(self) if self.meta.is_column() else f"({self})"
        expr_repr, exprs = self.parse()
        indices = expr_repr.extract_indices()
        # If no extra nodes (e.g. var.x.sum()), gp.quicksum is faster
        if len(self._nodes) == 0:

            def gurobi_vectorized_sum(series: Sequence[pl.Series]) -> Any:
                return gp.quicksum(series[indices[0]])

        # Optimization: Detect (var * coeff).sum() pattern and use gp.LinExpr directly
        # This is ~10x faster than element-wise multiplication followed by sum
        elif len(self._nodes) == 1 and self._nodes[0].operator in ("__mul__", "__rmul__"):

            def gurobi_vectorized_sum(series: Sequence[pl.Series]) -> Any:
                # If coeff are Object, fast path can't be taken
                if (len(indices) >= 2 and series[indices[1]].dtype is not pl.Object) or (
                    isinstance(self._nodes[0].operand, float | int)
                ):
                    operand = (
                        series[indices[1]].to_list()
                        if len(series) > 1
                        else len(series[indices[0]]) * [self._nodes[0].operand]
                    )
                    result = gp.LinExpr(operand, series[indices[0]].to_list())  # type: ignore

                else:
                    gp_obj = tuple(map(to_mvar_or_mlinexpr, series))
                    result = expr_repr.evaluate(gp_obj)
                return pl.Series([result], dtype=pl.Object)
        else:
            # Standard vectorized sum for other expression patterns
            def gurobi_vectorized_sum(series: Sequence[pl.Series]) -> Any:
                gp_obj = tuple(map(to_mvar_or_mlinexpr, series))
                result = expr_repr.evaluate(gp_obj)
                return pl.Series([result.sum().item()], dtype=pl.Object)

        _, exprs = self.parse()
        return self.__class__(
            pl.map_batches(
                exprs,
                gurobi_vectorized_sum,
                return_dtype=pl.Object,
                returns_scalar=True,
            ),
            name=name + ".sum()",
        )

    def any(self) -> Self:  # ty:ignore[invalid-method-override]
        """Create a Gurobi OR constraint from elements in each group.

        Parameters
        ----------
        expr : pl.Expr | str
            Column name or polars expression containing Gurobi variables or expressions

        Examples
        --------
        >>> df.group_by('group').agg(xmodel.var("x").any())

        """
        return self.__class__(
            self.map_batches(
                lambda series: gp.any_(*series),  # ty:ignore
                return_dtype=pl.Object,
                returns_scalar=True,
            ),
            name=f"{self}.any()",
        )

    def abs(self) -> Self:
        """Apply Gurobi's absolute value function to elements in each group.

        Parameters
        ----------
        expr : pl.Expr | str
            Column name or polars expression containing Gurobi variables or expressions

        Examples
        --------
        >>> df.with_columns(xmodel.var("x").abs())

        """
        return self.__class__(
            self.map_batches(
                lambda series: gp.abs_(*series),
                return_dtype=pl.Object,
                returns_scalar=True,
            ),
            name=f"{self}.abs()",
        )

    def all(self) -> Self:  # ty:ignore[invalid-method-override]
        """Create a Gurobi AND constraint from elements in each group.

        Returns 1 if all variables/expressions are non-zero (true), 0 otherwise.
        Equivalent to gp.and_() or gp.all_().

        Parameters
        ----------
        expr : pl.Expr | str
            Column name or polars expression containing Gurobi variables or expressions

        Examples
        --------
        >>> df.group_by('group').agg(xmodel.var("x").all())

        """
        return self.__class__(
            self.map_batches(
                lambda series: gp.and_(*series),
                return_dtype=pl.Object,
                returns_scalar=True,
            ),
            name=f"{self}.all()",
        )

    def max(self) -> Self:
        """Return the maximum of elements in each group.

        Parameters
        ----------
        expr : pl.Expr | str
            Column name or polars expression containing Gurobi variables or expressions

        Examples
        --------
        >>> df.group_by('group').agg(xmodel.var("x").max())

        """
        return self.__class__(
            self.map_batches(
                lambda series: gp.max_(*series),
                return_dtype=pl.Object,
                returns_scalar=True,
            ),
            name=f"{self}.max()",
        )

    def min(self) -> Self:
        """Return the minimum of elements in each group.

        Parameters
        ----------
        expr : pl.Expr | str
            Column name or polars expression containing Gurobi variables or expressions

        Examples
        --------
        >>> df.group_by('group').agg(xmodel.var("x").min())

        """
        return self.__class__(
            self.map_batches(
                lambda series: gp.min_(*series),
                return_dtype=pl.Object,
                returns_scalar=True,
            ),
            name=f"{self}.min()",
        )

    def _create_nl_func(self, func_name: str) -> Self:
        """Create a vectorized nonlinear function expression.

        Parameters
        ----------
        func_name : str
            Name of the Gurobi nlfunc function (e.g., 'square', 'exp', 'log').

        Returns
        -------
        Self
            New GurobiVarExpr with the nonlinear function applied.

        """
        nl_func = getattr(gurobipy.nlfunc, func_name)

        def vectorized_nl_func(series: pl.Series) -> pl.Series:
            mvar = to_mvar_or_mlinexpr(series)
            result = nl_func(mvar)
            # MNLExpr is iterable like MLinExpr - extract elements using .item()
            return pl.Series([r.item() for r in result], dtype=pl.Object)

        return self.__class__(
            self.map_batches(
                vectorized_nl_func,
                return_dtype=pl.Object,
                returns_scalar=False,
            ),
            name=f"{func_name}({self})",
        )

    def square(self) -> Self:
        """Square the variable (quadratic constraint, equivalent to x^2).

        Creates a quadratic expression using Gurobi's vectorized square function.
        This is much faster than element-wise squaring.

        Examples
        --------
        >>> df.with_columns(x_squared=xmodel.var("x").square())

        """
        return self._create_nl_func("square")

    def exp(self) -> Self:
        """Exponential function (e^x).

        Creates a nonlinear expression using Gurobi's vectorized exp function.

        Examples
        --------
        >>> df.with_columns(exp_x=xmodel.var("x").exp())

        """
        return self._create_nl_func("exp")

    def log(self) -> Self:  # ty:ignore[invalid-method-override]
        """Natural logarithm (ln(x)).

        Creates a nonlinear expression using Gurobi's vectorized log function.

        Examples
        --------
        >>> df.with_columns(log_x=xmodel.var("x").log())

        """
        return self._create_nl_func("log")

    def sin(self) -> Self:
        """Sine function.

        Creates a nonlinear expression using Gurobi's vectorized sin function.

        Examples
        --------
        >>> df.with_columns(sin_x=xmodel.var("x").sin())

        """
        return self._create_nl_func("sin")

    def cos(self) -> Self:
        """Cosine function.

        Creates a nonlinear expression using Gurobi's vectorized cos function.

        Examples
        --------
        >>> df.with_columns(cos_x=xmodel.var("x").cos())

        """
        return self._create_nl_func("cos")

    def sqrt(self) -> Self:
        """Square root function.

        Creates a nonlinear expression using Gurobi's vectorized sqrt function.

        Examples
        --------
        >>> df.with_columns(sqrt_x=xmodel.var("x").sqrt())

        """
        return self._create_nl_func("sqrt")

    def to_linexpr(self) -> Self:
        """Convert Gurobi variables to linear expressions.

        Transforms a column of gp.Var objects into gp.LinExpr objects.
        This is useful when you need to explicitly work with linear expressions
        instead of variables.

        Returns
        -------
        Self
            New GurobiVarExpr with variables converted to linear expressions.

        Examples
        --------
        >>> df.with_columns(x_as_linexpr=xmodel.var.x.to_linexpr())

        """

        def var_to_linexpr(series: pl.Series) -> pl.Series:
            return pl.Series(
                gp.MLinExpr._from_linexprs(series)._learr.tolist(),  # ty:ignore[unresolved-attribute]
                dtype=pl.Object,
            )

        return self.__class__(
            self.map_batches(
                var_to_linexpr,
                return_dtype=pl.Object,
                returns_scalar=False,
            ),
            name=str(self),
        )


class _ProxyGurobiVarExpr:
    def __call__(self, name: str, /, *more_names: str) -> GurobiVarExpr:
        return GurobiVarExpr(pl.col(name, *more_names))

    def __getattr__(self, name: str) -> GurobiVarExpr:
        return GurobiVarExpr(pl.col(name))
