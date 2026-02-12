from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import gurobipy as gp
import polars as pl

from xplor.exprs import VarExpr

if TYPE_CHECKING:
    from collections.abc import Sequence


def to_mvar_or_mlinexpr(s: pl.Series) -> gp.MVar | gp.MLinExpr | Any:
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
    first = next(iter(s))
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
        sum(): Vectorized sum using MVar/MLinExpr (~2-3x faster than element-wise).
        any(): Creates a Gurobi OR constraint across elements in each group.
        abs(): Applies Gurobi's absolute value function.
        _to_gp_expr(): Converts expression to vectorized Gurobi MVar/MLinExpr.

    """

    def _to_gp_expr(self, series: Sequence[pl.Series]) -> gp.MVar | gp.MLinExpr | Any:
        """Convert columns to MVar/MLinExpr and evaluate expression vectorized.

        This is the core Gurobi optimization: instead of evaluating row-by-row,
        we convert entire columns to MVar/MLinExpr arrays and evaluate once.

        Parameters
        ----------
        series : list[pl.DataFrame]
            List of series to evaluate.

        Returns
        -------
        gp.MVar | gp.MLinExpr | Any
            Vectorized Gurobi expression, or fallback result.

        Examples
        --------
        >>> expr = var.x + var.y  # GurobiVarExpr
        >>> result = expr._to_gp_expr(series)  # Returns single MLinExpr
        >>> summed = result.sum()  # Fast Gurobi sum

        """
        # For expressions without nodes, return as-is
        if not self._nodes:
            return gp.MVar(series[0])  # ty:ignore[too-many-positional-arguments]

        # Parse expression
        expr_repr, _ = self.parse()

        # Convert columns to MVar/MLinExpr arrays
        arrays = tuple(map(to_mvar_or_mlinexpr, series))

        # Vectorized evaluation - returns single MVar/MLinExpr
        return expr_repr.evaluate(arrays)

    def _to_expr(self) -> pl.Expr:
        if not self._nodes:
            return self._expr
        _, exprs = self.parse()

        def vectorized_eval(series: Sequence[pl.Series], **kwargs: Any) -> pl.Series:
            """Evaluate expression using vectorized NumPy operations."""
            result = self._to_gp_expr(series)

            if isinstance(result, gp.MLinExpr):
                # MLinExpr or NumPy array - convert to list
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

        For expressions with nodes (e.g., var.x + var.y), uses _to_gp_expr()
        to evaluate vectorized, then calls .sum() on the result. This is
        2-3x faster than element-wise summation.

        Optimized pattern: (var * coeff).sum() uses gp.LinExpr(coeffs, vars)
        directly, which is ~10x faster than element-wise multiplication + sum.

        Examples
        --------
        >>> df.group_by('group').agg(xmodel.var("x").sum())
        >>> df.select(total=(var.x + var.y).sum())  # Vectorized!
        >>> df.select(total=(var.x * pl.col("cost")).sum())  # Optimized LinExpr!

        """
        name = str(self) if self.meta.is_column() else f"({self})"
        _, exprs = self.parse()
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
                    result = self._to_gp_expr(series)
                return pl.Series([result], dtype=pl.Object)
        else:
            # Standard vectorized sum for other expression patterns
            def gurobi_vectorized_sum(series: Sequence[pl.Series]) -> Any:
                result = self._to_gp_expr(series)
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
                lambda d: gp.or_(d.to_list()), return_dtype=pl.Object, returns_scalar=True
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
            self.map_elements(lambda d: gp.abs_(d), return_dtype=pl.Object),
            name=f"{self}.abs()",
        )


class _ProxyGurobiVarExpr:
    def __call__(self, name: str, /, *more_names: str) -> GurobiVarExpr:
        return GurobiVarExpr(pl.col(name, *more_names))

    def __getattr__(self, name: str) -> GurobiVarExpr:
        return GurobiVarExpr(pl.col(name))
