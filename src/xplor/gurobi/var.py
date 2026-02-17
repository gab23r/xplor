from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import gurobipy as gp
import gurobipy.nlfunc
import polars as pl

from xplor.exprs import VarExpr

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


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
                if result._learr is not None:
                    result = result._learr.tolist()
                else:
                    result = list(map(lambda r: r.item(), result))  # noqa: C417
            return pl.Series(result, dtype=pl.Object)

        return pl.map_batches(
            exprs,
            vectorized_eval,
            return_dtype=pl.Object,
        )

    def sum_by(
        self,
        by: str | pl.Expr | list[str],
    ) -> Self:
        """Compute grouped sum using Gurobi's matrix API for optimal performance.

        Uses Gurobi's matrix API for efficient grouped aggregations via sparse
        matrix multiplication. Returns one row per group, matching Polars'
        groupby semantics.

        Parameters
        ----------
        by : str | pl.Expr | list[str]
            Column(s) to group by. Accepts:
            - String: column name (e.g., "w")
            - pl.Expr: expression (e.g., pl.col("w"))
            - list[str]: multiple columns (e.g., ["w", "region"])

        Returns
        -------
        Self
            One row per group with group sums

        Examples
        --------
        >>> # Grouped sum - returns one row per group
        >>> df.select(group_sum=xmodel.var.x.sum_by("w"))

        >>> # Weighted grouped sum
        >>> df.select((xmodel.var.x * pl.col("cost")).sum_by("w"))

        >>> # Multi-column groupby
        >>> df.select(xmodel.var.x.sum_by(["w", "region"]))

        >>> # Use in constraints
        >>> df_grouped = df.select(sum=xmodel.var.x.sum_by("w"))
        >>> xmodel.add_constrs(df_grouped, limit=xplor.var.sum <= 500)

        """
        # Convert string or list of strings to pl.Expr
        if isinstance(by, str):
            group_by = pl.col(by)
        elif isinstance(by, list):
            group_by = pl.struct(by)
        else:
            group_by = by
        import numpy as np
        import scipy.sparse as sp

        name = str(self) if self.meta.is_column() else f"({self})"
        expr_repr, exprs = self.parse()

        # Add the group_by expression to the list of expressions to evaluate
        exprs_with_group = [*exprs, group_by]

        # Build function that uses sparse matrix for groupby
        def gurobi_grouped_sum(series: Sequence[pl.Series], **kwargs: Any) -> pl.Series:
            # Last series is the group column
            group_series = series[-1]
            var_series_list = series[:-1]

            # Map unique group values to group IDs
            # Waiting for: https://github.com/pola-rs/polars/issues/25382
            unique_groups = group_series.unique(maintain_order=True)
            group_ids = (
                group_series.to_frame("by")
                .join(
                    unique_groups.to_frame("by").with_row_index(), on="by", maintain_order="left"
                )["index"]
                .to_numpy()
            )
            n_groups = len(unique_groups)
            n_vars = len(series[0])

            # Build sparse matrix for groupby aggregation
            col = np.arange(n_vars)
            row = group_ids
            val = np.ones(n_vars)
            A = sp.csr_matrix((val, (row, col)), shape=(n_groups, n_vars))

            gp_obj = tuple(map(to_mvar_or_mlinexpr, var_series_list))
            result = expr_repr.evaluate(gp_obj)
            group_sums = A @ result

            if group_sums._learr is not None:
                result = group_sums._learr.tolist()
            else:
                result = list(map(lambda r: r.item(), group_sums))  # noqa: C417

            # Return group-level results (one per group, not broadcast)
            return pl.Series(result, dtype=pl.Object)

        return self.__class__(
            pl.map_batches(
                exprs_with_group,
                gurobi_grouped_sum,
                return_dtype=pl.Object,
                returns_scalar=False,
            ),
            name=name + f".sum_by({by})",
        )

    def sum(self) -> Self:
        """Vectorized sum using Gurobi MVar/MLinExpr for optimal performance.

        Optimized pattern: (var * coeff).sum() uses gp.LinExpr(coeffs, vars)
        directly, which is much faster than element-wise multiplication + sum.

        Returns
        -------
        Self
            Scalar aggregation (single sum)

        Examples
        --------
        >>> # Regular sum
        >>> df.select(total=xmodel.var.x.sum())

        >>> # Weighted sum
        >>> df.select(total=(xmodel.var.x * pl.col("cost")).sum())

        >>> # For grouped aggregations, use sum_by()
        >>> df.select(group_sum=xmodel.var.x.sum_by("w"))

        """
        # Implementation for non-grouped sum
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
