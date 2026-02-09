from __future__ import annotations

from typing import Self

import polars as pl

from xplor.exprs import ConstrExpr
from xplor.exprs.obj import ObjExpr


class VarExpr(ObjExpr):
    """A specialized custom Polars Expression wrapper, extending ObjExpr,
    designed for constructing composite expressions or linear expressions.

    Methods:
        sum(): Calculates the sum of optimization objects (e.g., for objective function creation).
        any(): Creates a Gurobi OR constraint across elements in each group.
        abs(): Applies Gurobi's absolute value function.

    """

    def sum(
        self,
    ) -> Self:
        """Get sum value.

        Examples
        --------
        >>> df.group_by('group').agg(xplor.var.sum())

        """
        name = str(self) if self.meta.is_column() else f"({self})"
        return self.__class__(
            self.map_batches(
                lambda d: sum(d.to_list()), return_dtype=pl.Object, returns_scalar=True
            ),
            name=name + ".sum()",
        )

    @property
    def name(self) -> VarExprNameNameSpace:
        """Create an object namespace of all var expressions that modify expression names."""
        return VarExprNameNameSpace(self)

    def __eq__(self, other: pl.Expr | float) -> ConstrExpr:
        return ConstrExpr.from_obj_expr(self._append_node("__eq__", other))

    def __le__(self, other: pl.Expr | float) -> ConstrExpr:
        return ConstrExpr.from_obj_expr(self._append_node("__le__", other))

    def __ge__(self, other: pl.Expr | float) -> ConstrExpr:
        return ConstrExpr.from_obj_expr(self._append_node("__ge__", other))


class VarExprNameNameSpace:
    """Namespace for var expressions that operate on expression names."""

    _accessor = "name"

    def __init__(self, expr: VarExpr) -> None:
        self._pyexpr = expr._pyexpr

    def prefix(self, prefix: str) -> VarExpr:
        """Add a prefix to the root column name of the object expression."""
        return VarExpr(pl.Expr._from_pyexpr(self._pyexpr.name_prefix(prefix)))

    def suffix(self, suffix: str) -> VarExpr:
        """Add a suffix to the root column name of the object expression."""
        return VarExpr(pl.Expr._from_pyexpr(self._pyexpr.name_suffix(suffix)))


class _ProxyVarExpr:
    """The entry point for creating custom expression objects (VarExpr) that represent
    variables or columns used within a composite Polars expression chain.

    This proxy acts similarly to `polars.col()`, allowing you to reference
    optimization variables (created via `xmodel.add_vars()`) or standard DataFrame columns
    in a solver-compatible expression.

    The resulting expression object can be combined with standard Polars expressions
    to form constraints or objective function components.

    Examples
    --------
    >>> xmodel = XplorMathOpt()
    >>> df = df.with_columns(xmodel.add_vars("production"))
    >>> df.select(total_cost = xplor.var("production") * pl.col("cost"))

    """

    def __call__(self, name: str, /, *more_names: str) -> VarExpr:
        """Create an ObjExpr instance using the call syntax: `var("column_name")`.

        This method is typically used when the variable name contains characters
        that are invalid for Python identifiers (e.g., spaces, special characters).

        Args:
            name: The string name of the variable or column.
            more_names: Other names.

        Returns:
            An ObjExpr object initialized with the given name.

        """
        return VarExpr(pl.col(name, *more_names))

    def __getattr__(self, name: str) -> VarExpr:
        """Create an ObjExpr instance using attribute access syntax: `var.column_name`.

        This provides a cleaner, more Pythonic syntax when the variable name is
        a valid Python identifier.

        Args:
            name: The string name of the variable or column (inferred from the attribute access).

        Returns:
            An ObjExpr object initialized with the attribute name.

        """
        return VarExpr(pl.col(name))
