from __future__ import annotations

from typing import Self

import gurobipy as gp
import polars as pl

from xplor.exprs import VarExpr


class GurobiVarExpr(VarExpr):
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
        >>> df.group_by('group').agg(xmodel.var("x").sum())

        """
        name = str(self) if self.meta.is_column() else f"({self})"
        return self.__class__(
            self.map_batches(
                lambda d: gp.quicksum(d.to_list()), return_dtype=pl.Object, returns_scalar=True
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
