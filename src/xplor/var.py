from __future__ import annotations

from enum import Enum

import polars as pl

from xplor.obj_expr import ObjExpr


class VarType(str, Enum):
    """The type of the variable."""

    CONTINUOUS = "CONTINUOUS"
    INTEGER = "INTEGER"
    BINARY = "BINARY"


class VarExpr(ObjExpr):
    """A specialized custom Polars Expression wrapper, extending ObjExpr,
    designed for constructing composite expressions or linear expressions.

    Methods:
        sum(): Calculates the sum of optimization objects (e.g., for objective function creation).
        any(): Creates a Gurobi OR constraint across elements in each group.
        abs(): Applies Gurobi's absolute value function.
        read_value(): Extracts the optimal solution value (X or getValue()) after model solving.

    """

    def sum(
        self,
    ) -> pl.Expr:
        """Get sum value.

        Examples
        --------
        >>> df.group_by('group').agg(xpl.var.sum())

        """
        return self.map_batches(lambda d: sum(d), return_dtype=pl.Object, returns_scalar=True)

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

        return self.map_batches(
            lambda d: gp.or_(d.to_list()), return_dtype=pl.Object, returns_scalar=True
        )

    def abs(self) -> pl.Expr:
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

        return self.map_elements(lambda d: gp.abs_(d), return_dtype=pl.Object)

    def read_value(self) -> pl.Expr:
        """Extract the optimal value from variables or expressions after optimization.

        Parameters
        ----------
        expr : str
            Column name or polars expression containing Gurobi variables or linear expressions

        Returns
        -------
        pl.Expr
            Expression that will return the optimal values after model solving.
            For variables, returns X attribute value.
            For linear expressions, returns the evaluated value.

        Examples
        --------
        >>> df.with_columns(xpl.var.x.read_value())

        """
        return self.map_batches(
            lambda s:
            # in case of a variable
            pl.Series([e.x for e in s])
            if s.len() and hasattr(s[0], "X")
            # in case of a linExpr
            else pl.Series([e.getValue() for e in s]),
            return_dtype=pl.Float64,
        )


class ProxyObjExpr:
    """The entry point for creating custom expression objects (ObjExpr) that represent
    variables or columns used within a composite Polars expression chain.
    """

    def __call__(
        self,
        name: str,
    ) -> VarExpr:
        """Create an ObjExpr instance using the call syntax: `var("column_name")`.

        This method is typically used when the variable name contains characters
        that are invalid for Python identifiers (e.g., spaces, special characters).

        Args:
            name: The string name of the variable or column.

        Returns:
            An ObjExpr object initialized with the given name.

        """
        return VarExpr(name)

    def __getattr__(self, name: str) -> VarExpr:
        """Create an ObjExpr instance using attribute access syntax: `var.column_name`.

        This provides a cleaner, more Pythonic syntax when the variable name is
        a valid Python identifier.

        Args:
            name: The string name of the variable or column (inferred from the attribute access).

        Returns:
            An ObjExpr object initialized with the attribute name.

        """
        return VarExpr(name)


var = ProxyObjExpr()
