from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import gurobipy as gp
import polars as pl

from xplor.exprs import VarExpr

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def to_mvar_or_mlinexpr(s: pl.Series) -> gp.MVar | gp.MLinExpr | np.ndarray:
    """Convert a Polars Series to Gurobi MVar, MLinExpr, or NumPy array.

    Parameters
    ----------
    s : pl.Series
        Series containing Gurobi variables, linear expressions, or numeric values.

    Returns
    -------
    gp.MVar | gp.MLinExpr | np.ndarray
        Vectorized Gurobi array or NumPy array.

    """
    first: Any = s.first(ignore_nulls=True)
    if isinstance(first, gp.Var):
        return gp.MVar(s)  # ty:ignore[too-many-positional-arguments]
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

    """

    def _to_expr(self) -> pl.Expr:
        if not self._nodes:
            return self._expr
        expr_repr, exprs = self.parse()

        def vectorized_eval(series: Sequence[pl.Series], **kwargs: Any) -> pl.Series:
            """Evaluate expression using vectorized NumPy operations."""
            gp_obj = tuple(map(to_mvar_or_mlinexpr, series))
            result = expr_repr.evaluate(gp_obj)

            # Unfortunalty `MLinExpr` need to be disptach into list of LinExpr from the python side.
            if isinstance(result, gp.MLinExpr):
                result = [r.item() for r in result]
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
        # Optimization: Detect (var * coeff).sum() pattern and use gp.LinExpr directly
        # This is ~10x faster than element-wise multiplication followed by sum
        if len(self._nodes) == 1 and self._nodes[0].operator in ("__mul__", "__rmul__"):

            def gurobi_vectorized_sum(series: Sequence[pl.Series]) -> Any:
                # If coeff are Object, fast path can't be taken
                if (len(series) >= 2 and series[1].dtype is not pl.Object) or (
                    isinstance(self._nodes[0].operand, float)
                ):
                    operand = (
                        series[1].to_list()
                        if len(series) > 1
                        else len(series[0]) * [self._nodes[0].operand]
                    )
                    result = gp.LinExpr(operand, series[0].to_list())  # type: ignore

                else:
                    gp_obj = tuple(map(to_mvar_or_mlinexpr, series))
                    result = expr_repr.evaluate(gp_obj)
                return pl.Series([result], dtype=pl.Object)
        else:
            # Standard vectorized sum for other expression patterns
            def gurobi_vectorized_sum(series: Sequence[pl.Series]) -> Any:
                gp_obj = tuple(map(to_mvar_or_mlinexpr, series))
                result = expr_repr.evaluate(gp_obj)
                return pl.Series([result.sum()], dtype=pl.Object)

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


class _ProxyGurobiVarExpr:
    def __call__(self, name: str, /, *more_names: str) -> GurobiVarExpr:
        return GurobiVarExpr(pl.col(name, *more_names))

    def __getattr__(self, name: str) -> GurobiVarExpr:
        return GurobiVarExpr(pl.col(name))
