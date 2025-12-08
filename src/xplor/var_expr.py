from __future__ import annotations

import polars as pl

from xplor.obj_expr import ObjExpr


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
    ) -> VarExpr:
        """Get sum value.

        Examples
        --------
        >>> df.group_by('group').agg(xpl.var.sum())

        """
        name = str(self) if self.meta.is_column() else f"({self})"
        return VarExpr(
            self.map_batches(lambda d: sum(d), return_dtype=pl.Object, returns_scalar=True),
            name=name + ".sum()",
        )

    def any(self) -> pl.Expr:  # type: ignore
        """Create a Gurobi OR constraint from elements in each group.

        Parameters
        ----------
        expr : pl.Expr | str
            Column name or polars expression containing Gurobi variables or expressions

        Returns
        -------
        pl.Expr
            Expression that will return the Gurobi OR of elements in each group

        Examples
        --------
        >>> df.group_by('group').agg(xpl.var.any())

        """
        import gurobipy as gp

        return VarExpr(
            self.map_batches(
                lambda d: gp.or_(d.to_list()), return_dtype=pl.Object, returns_scalar=True
            ),
            name=f"{self}.any()",
        )

    def abs(self) -> VarExpr:
        """Apply Gurobi's absolute value function to elements in each group.

        Parameters
        ----------
        expr : pl.Expr | str
            Column name or polars expression containing Gurobi variables or expressions

        Returns
        -------
        pl.Expr
            Expression that will return the absolute value of elements in each group

        Examples
        --------
        >>> df.with_columns(xpl.var.abs())

        """
        import gurobipy as gp

        return VarExpr(
            self.map_elements(lambda d: gp.abs_(d), return_dtype=pl.Object),
            name=f"{self}.abs()",
        )


class _ProxyObjExpr:
    """The entry point for creating custom expression objects (VarExpr) that represent
    variables or columns used within a composite Polars expression chain.

    This proxy acts similarly to `polars.col()`, allowing you to reference
    optimization variables (created via `xmodel.add_vars()`) or standard DataFrame columns
    in a solver-compatible expression.

    The resulting expression object can be combined with standard Polars expressions
    to form constraints or objective function components.

    Examples:
    ```python

    >>> xmodel = XplorMathOpt()
    >>> df = df.with_columns(xmodel.add_vars("production"))
    >>> df.select(total_cost = xplor.var("production") * pl.col("cost"))
    ```

    """

    def __call__(self, name: str, /) -> VarExpr:
        """Create an ObjExpr instance using the call syntax: `var("column_name")`.

        This method is typically used when the variable name contains characters
        that are invalid for Python identifiers (e.g., spaces, special characters).

        Args:
            name: The string name of the variable or column.

        Returns:
            An ObjExpr object initialized with the given name.

        """
        return VarExpr(pl.col(name))

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
